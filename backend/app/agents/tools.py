"""
Read-only analysis tools for the Analyst agent. Every result is registered in the facts store so answers
can cite it ([T1], [F3]) and the critic can verify the numbers.

SQL runs in an in-memory DuckDB with external access disabled and configuration locked: queries can only
see the registered `data` table; files, network, COPY/ATTACH/SET are rejected by DuckDB itself.
"""
import math
import re
from typing import Any, Dict, List

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as sps

from app.agents.context import redact
from app.agents.facts import FactStore
from app.core.config import settings

MAX_RESULT_ROWS = 200
OBS_ROWS = 25
CHART_TYPES = {"bar", "line", "scatter"}
_FORBIDDEN = re.compile(r"\b(copy|attach|detach|install|load|pragma|export|import|create|insert|update|delete|drop|alter|call|set|reset|checkpoint|vacuum)\b", re.I)


class ToolError(ValueError):
    pass


def _cell(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return str(v)[:120]


class DataTools:
    def __init__(self, df: pd.DataFrame, facts: FactStore):
        self.df = df
        self.facts = facts
        self.charts: List[Dict[str, Any]] = []
        self.con = duckdb.connect(":memory:", config={"enable_external_access": False})
        self.con.register("data", df)
        self.con.execute("SET lock_configuration=true")

    # ── catalogue shown to the model ──
    SPEC = [
        {"name": "schema", "args": {}, "does": "List columns with types, missing % and example values."},
        {"name": "sql", "args": {"query": "a single DuckDB SELECT/WITH over table data"},
         "does": "Run a read-only query. Quote column names with double quotes. Returns up to 200 rows as table T#."},
        {"name": "stat_test", "args": {"kind": "compare_groups | correlation | association", "a": "column", "b": "column"},
         "does": "compare_groups: numeric a across groups of b (Mann-Whitney / Kruskal). correlation: numeric a vs b (Spearman). "
                 "association: categorical a vs b (chi-square, Cramér's V)."},
        {"name": "chart", "args": {"table": "T#", "type": "bar | line | scatter", "x": "column of that table",
                                   "y": "column or list of columns", "title": "short title"},
         "does": "Create a chart from a previous sql result table."},
    ]

    def run(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "schema":
            return self.schema()
        if name == "sql":
            return self.sql(str(args.get("query", "")))
        if name == "stat_test":
            return self.stat_test(str(args.get("kind", "")), str(args.get("a", "")), str(args.get("b", "")))
        if name == "chart":
            return self.chart(args)
        raise ToolError(f"Unknown tool '{name}'.")

    def schema(self) -> Dict[str, Any]:
        cols = []
        for c in self.df.columns:
            s = self.df[c]
            entry = {"name": c, "dtype": str(s.dtype), "missing_pct": round(float(s.isna().mean() * 100), 1), "unique": int(s.nunique())}
            if settings.AI_PRIVACY_MODE == "samples":
                entry["examples"] = [redact(v) for v in s.dropna().drop_duplicates().head(3).tolist()]
            cols.append(entry)
        return {"table": "data", "rows": int(len(self.df)), "columns": cols}

    def sql(self, query: str) -> Dict[str, Any]:
        q = query.strip().rstrip(";").strip()
        if not q:
            raise ToolError("Empty query.")
        if ";" in q:
            raise ToolError("Only one statement is allowed.")
        if not re.match(r"^(select|with)\b", q, re.I):
            raise ToolError("Only SELECT / WITH queries are allowed.")
        if _FORBIDDEN.search(re.sub(r"'[^']*'", "''", q)):
            raise ToolError("The query contains a forbidden keyword.")
        try:
            res = self.con.execute(f"SELECT * FROM ({q}) AS _q LIMIT {MAX_RESULT_ROWS}").fetchdf()
        except Exception as e:
            raise ToolError(f"SQL error: {str(e).splitlines()[0][:300]}")
        columns = [str(c) for c in res.columns]
        rows = [[_cell(v) for v in row] for row in res.itertuples(index=False, name=None)]
        tid = self.facts.add_table(q[:160], columns, rows, source="sql")
        return {"table_id": tid, "columns": columns, "row_count": len(rows), "rows": rows[:OBS_ROWS],
                "truncated": len(rows) > OBS_ROWS}

    def _col(self, name: str) -> pd.Series:
        if name not in self.df.columns:
            raise ToolError(f"Column '{name}' does not exist. Call schema to see the exact names.")
        return self.df[name]

    def stat_test(self, kind: str, a: str, b: str) -> Dict[str, Any]:
        x, y = self._col(a), self._col(b)
        data = pd.DataFrame({"a": x, "b": y}).dropna()
        if len(data) < 8:
            raise ToolError("Not enough non-missing rows for a test.")
        if kind == "compare_groups":
            vals = pd.to_numeric(data["a"], errors="coerce")
            data = data[vals.notna()].assign(a=vals[vals.notna()])
            groups = data.groupby("b")["a"]
            sizes = groups.size().sort_values(ascending=False).head(12)
            samples = [data.loc[data["b"] == g, "a"] for g in sizes.index]
            if len(samples) < 2:
                raise ToolError("Need at least two groups.")
            if len(samples) == 2:
                stat, p = sps.mannwhitneyu(samples[0], samples[1])
                test = "Mann-Whitney U"
            else:
                stat, p = sps.kruskal(*samples)
                test = "Kruskal-Wallis"
            rows = [[str(g), int(len(s)), round(float(s.mean()), 6), round(float(s.median()), 6)] for g, s in zip(sizes.index, samples)]
            tid = self.facts.add_table(f"{a} by {b}", [str(b), "n", f"mean {a}", f"median {a}"], rows, source="stat_test")
            fid = self.facts.add(f"{test} p-value for {a} across {b}", round(float(p), 6), source="stat_test")
            return {"test": test, "p_value": round(float(p), 6), "p_fact": fid, "table_id": tid, "groups": rows}
        if kind == "correlation":
            xa, ya = pd.to_numeric(data["a"], errors="coerce"), pd.to_numeric(data["b"], errors="coerce")
            ok = xa.notna() & ya.notna()
            if ok.sum() < 8:
                raise ToolError("Both columns must be numeric.")
            rho, p = sps.spearmanr(xa[ok], ya[ok])
            r_fact = self.facts.add(f"Spearman correlation of {a} and {b}", round(float(rho), 6), source="stat_test")
            p_fact = self.facts.add(f"p-value of that correlation", round(float(p), 6), source="stat_test")
            return {"spearman": round(float(rho), 6), "p_value": round(float(p), 6), "facts": [r_fact, p_fact], "n": int(ok.sum())}
        if kind == "association":
            table = pd.crosstab(data["a"].astype(str), data["b"].astype(str))
            if table.shape[0] > 50 or table.shape[1] > 50:
                raise ToolError("Too many categories for a chi-square test (max 50 per side).")
            chi2, p, dof, _ = sps.chi2_contingency(table)
            n = table.values.sum()
            v = math.sqrt(chi2 / (n * max(1, min(table.shape) - 1)))
            v_fact = self.facts.add(f"Cramér's V between {a} and {b}", round(v, 6), source="stat_test")
            p_fact = self.facts.add("chi-square p-value", round(float(p), 6), source="stat_test")
            return {"cramers_v": round(v, 6), "p_value": round(float(p), 6), "facts": [v_fact, p_fact]}
        raise ToolError("kind must be compare_groups, correlation or association.")

    def chart(self, args: Dict[str, Any]) -> Dict[str, Any]:
        table = self.facts.table(str(args.get("table", "")))
        if not table:
            raise ToolError("Unknown table id; run sql first and use its table_id.")
        ctype = str(args.get("type", "bar")).lower()
        if ctype not in CHART_TYPES:
            raise ToolError("type must be bar, line or scatter.")
        x = str(args.get("x", ""))
        ys = args.get("y")
        ys = [str(v) for v in (ys if isinstance(ys, list) else [ys])] if ys else []
        missing = [c for c in [x, *ys] if c not in table["columns"]]
        if missing or not ys:
            raise ToolError(f"Columns {missing or ['y']} are not in {table['id']} (available: {table['columns']}).")
        cid = f"C{len(self.charts) + 1}"
        self.charts.append({"id": cid, "type": ctype, "title": str(args.get("title") or "")[:120], "table": table["id"],
                            "x": x, "y": ys, "columns": table["columns"], "rows": table["rows"][:MAX_RESULT_ROWS]})
        return {"chart_id": cid, "type": ctype, "points": min(len(table["rows"]), MAX_RESULT_ROWS)}
