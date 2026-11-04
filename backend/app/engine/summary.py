"""
Dataset summary: everything the summary page charts, computed on the full data, plus plain-language insights.

Combines the explore payload (column stats, correlations, PCA projection) with full-data histograms, date timelines,
missingness, target drivers (with per-group breakdowns), and the stored pipeline results (health score, issues,
cleaning, model leaderboard, agent report). Every insight sentence is generated from computed numbers.
"""
import math
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.engine.stats import target_association
from app.engine.visualizer import build_explore_payload

MAX_HIST_BINS = 30
MAX_HEATMAP_COLS = 12
MAX_DRIVERS = 8
MAX_TIMELINE_POINTS = 60


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _hist(values: pd.Series) -> Optional[Dict[str, Any]]:
    v = values.dropna()
    if len(v) < 2 or v.nunique() < 2:
        return None
    bins = int(min(MAX_HIST_BINS, max(8, math.ceil(math.log2(len(v)) + 1) * 2)))
    if v.nunique() <= bins:
        counts = v.value_counts().sort_index()
        return {"discrete": True, "values": [float(x) for x in counts.index], "counts": [int(c) for c in counts]}
    counts, edges = np.histogram(v, bins=bins)
    return {"discrete": False, "edges": [round(float(e), 6) for e in edges], "counts": [int(c) for c in counts]}


def _box(values: pd.Series) -> Optional[Dict[str, float]]:
    v = values.dropna()
    if len(v) < 4:
        return None
    q1, med, q3 = (float(v.quantile(q)) for q in (0.25, 0.5, 0.75))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = v[(v >= lo) & (v <= hi)]
    return {"q1": q1, "median": med, "q3": q3, "lo": float(inside.min()), "hi": float(inside.max()),
            "min": float(v.min()), "max": float(v.max()), "outliers": int(((v < lo) | (v > hi)).sum()),
            "extreme": int(((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)).sum())}


def _timeline(series: pd.Series) -> Optional[Dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(series, errors="coerce", format="mixed", utc=True).dropna()
    if len(parsed) < 5 or parsed.nunique() < 2:
        return None
    span = (parsed.max() - parsed.min()).days
    freq, label = (("YS", "year") if span > 365 * 15 else ("QS", "quarter") if span > 365 * 5 else
                   ("MS", "month") if span > 90 else ("W", "week") if span > 21 else ("D", "day"))
    counts = parsed.dt.tz_localize(None).to_frame("t").set_index("t").resample(freq).size()
    if len(counts) > MAX_TIMELINE_POINTS:
        counts = counts.iloc[-MAX_TIMELINE_POINTS:]
    return {"granularity": label, "points": [{"t": t.isoformat(), "count": int(c)} for t, c in counts.items()],
            "start": parsed.min().isoformat(), "end": parsed.max().isoformat(), "span_days": int(span)}


def _driver_breakdown(df: pd.DataFrame, col: str, target: str, problem_type: str) -> Optional[Dict[str, Any]]:
    """Target rate / mean per group of the driver (top categories, or quantile bins for numbers)."""
    data = df[[col, target]].dropna()
    if len(data) < 10:
        return None
    x = data[col]
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > 8:
        try:
            groups = pd.qcut(x, q=5, duplicates="drop").astype(str)
        except ValueError:
            return None
    else:
        top = x.astype(str).value_counts().head(8).index
        groups = x.astype(str).where(x.astype(str).isin(top), "other")
    y = data[target]
    if problem_type == "classification":
        classes = y.astype(str).value_counts()
        focus = classes.index[-1] if len(classes) == 2 else classes.index[0]
        rate = (y.astype(str) == focus).groupby(groups).mean()
        n = groups.value_counts()
        return {"measure": f"share of '{focus}'", "focus": str(focus), "overall": round(float((y.astype(str) == focus).mean()), 4),
                "groups": [{"group": str(g), "value": round(float(rate[g]), 4), "n": int(n[g])} for g in rate.sort_values(ascending=False).index]}
    yn = _num(y)
    means = yn.groupby(groups).mean()
    n = groups.value_counts()
    return {"measure": f"mean {target}", "overall": round(float(yn.mean()), 4),
            "groups": [{"group": str(g), "value": round(float(means[g]), 4), "n": int(n[g])} for g in means.sort_values(ascending=False).index]}


def _insight(kind: str, tone: str, title: str, text: str, **extra) -> Dict[str, Any]:
    return {"kind": kind, "tone": tone, "title": title, "text": text, **extra}


def build_summary(
    df: pd.DataFrame,
    target: Optional[str],
    problem_type: Optional[str],
    diagnostics: Optional[Dict[str, Any]] = None,
    execution: Optional[Dict[str, Any]] = None,
    leaderboard: Optional[Dict[str, Any]] = None,
    report: Optional[Dict[str, Any]] = None,
    assignments: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    explore = build_explore_payload(df, target=target, max_rows=1200, max_cols=80)
    n_rows, n_cols = len(df), df.shape[1]
    columns = []
    kinds: Dict[str, int] = {}
    for c in explore["columns"]:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        s = df[c["name"]]
        entry = {k: c.get(k) for k in ("name", "kind", "inferred_type", "is_target", "missing_pct", "unique", "stats", "range")}
        if c["kind"] == "numeric":
            entry["hist"], entry["box"] = _hist(_num(s)), _box(_num(s))
        elif c["kind"] == "datetime":
            entry["timeline"] = _timeline(s)
        else:
            counts = s.astype("string").value_counts()
            top = counts.head(10)
            entry["top"] = [{"value": str(v), "count": int(k)} for v, k in top.items()]
            entry["other_count"] = int(counts.iloc[10:].sum()) if len(counts) > 10 else 0
            entry["top_share"] = round(float(counts.iloc[0] / max(1, counts.sum())), 4) if len(counts) else 0.0
        columns.append(entry)

    # Missingness
    miss = df.isna()
    missing = {
        "cells_pct": round(float(miss.to_numpy().mean() * 100), 2) if n_rows and n_cols else 0.0,
        "complete_rows_pct": round(float((~miss.any(axis=1)).mean() * 100), 2) if n_rows else 0.0,
        "by_column": sorted([{"column": str(c), "pct": round(float(miss[c].mean() * 100), 2), "count": int(miss[c].sum())}
                             for c in df.columns if miss[c].any()], key=lambda r: -r["pct"]),
    }
    duplicates = int(df.duplicated().sum())

    # Correlations: compact heatmap + strongest pairs
    corr = explore.get("correlation")
    pairs: List[Dict[str, Any]] = []
    heat = None
    if corr:
        cols = corr["columns"]
        m = corr["pearson"]
        for i, a in enumerate(cols):
            for j in range(i + 1, len(cols)):
                r = m[i][j]
                if r is not None and not (isinstance(r, float) and math.isnan(r)):
                    pairs.append({"a": a, "b": cols[j], "r": round(float(r), 3)})
        pairs.sort(key=lambda p: -abs(p["r"]))
        keep = cols[:MAX_HEATMAP_COLS]
        idx = [cols.index(c) for c in keep]
        heat = {"columns": keep, "matrix": [[m[i][j] for j in idx] for i in idx]}

    # Target analysis
    target_info = None
    if target and target in df.columns:
        y = df[target]
        target_info = {"name": target, "problem_type": problem_type, "missing": int(y.isna().sum())}
        if problem_type == "classification":
            counts = y.astype("string").value_counts()
            target_info["classes"] = [{"value": str(v), "count": int(k)} for v, k in counts.head(15).items()]
        else:
            target_info["hist"], target_info["box"] = _hist(_num(y)), _box(_num(y))
        drivers = []
        for c in df.columns:
            if c == target:
                continue
            a = target_association(df, str(c), target, problem_type or "classification")
            if a is not None:
                drivers.append({"feature": str(c), "association": a})
        drivers.sort(key=lambda d: -d["association"])
        drivers = drivers[:MAX_DRIVERS]
        for d in drivers[:3]:
            d["breakdown"] = _driver_breakdown(df, d["feature"], target, problem_type or "classification")
        target_info["drivers"] = drivers

    # Projection (PCA of numeric features) with colour groups
    proj = explore.get("projection")
    projection = None
    if proj:
        sample_idx = None
        color, color_label = None, None
        if assignments is not None and len(assignments) == n_rows:
            sample_rows = df.sample(min(1200, n_rows), random_state=42).sort_index().index if n_rows > 1200 else df.index
            sample_idx = list(sample_rows)
            key = "cluster" if "cluster" in assignments.columns else "is_anomaly" if "is_anomaly" in assignments.columns else None
            if key:
                vals = assignments[key].reset_index(drop=True)
                pos = [df.index.get_loc(i) for i in sample_idx]
                color = [("noise" if key == "cluster" and int(vals[p]) == -1 else f"cluster {int(vals[p]) + 1}") if key == "cluster"
                         else ("anomaly" if int(vals[p]) == 1 else "normal") for p in pos]
                color_label = "cluster (benchmark)" if key == "cluster" else "anomaly flag"
        if color is None and target and problem_type == "classification":
            color = explore["data"].get(target)
            color_label = target
        if color is None and proj.get("clusters"):
            color, color_label = proj["clusters"], f"k-means (k={proj.get('k')})"
        projection = {"coords": proj["coords"], "explained": proj["explained"], "loadings": proj["loadings"],
                      "features": proj["features"], "color": color, "color_label": color_label}

    # Insights
    ins: List[Dict[str, Any]] = []
    numeric = [c for c in columns if c["kind"] == "numeric" and c.get("stats")]
    ins.append(_insight("shape", "info", f"{n_rows:,} rows × {n_cols} columns",
                        ", ".join(f"{v} {k}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])) + " column(s)."))
    if missing["by_column"]:
        worst = missing["by_column"][0]
        tone = "bad" if worst["pct"] > 30 else "warn" if worst["pct"] > 5 else "info"
        ins.append(_insight("missing", tone, f"{missing['cells_pct']}% of cells are blank",
                            f"{len(missing['by_column'])} column(s) have gaps; worst is '{worst['column']}' at {worst['pct']}%. "
                            f"{missing['complete_rows_pct']}% of rows are fully complete.", column=worst["column"]))
    else:
        ins.append(_insight("missing", "good", "No missing values", "Every cell has a value."))
    if duplicates:
        ins.append(_insight("duplicates", "warn" if duplicates / max(n_rows, 1) > 0.01 else "info",
                            f"{duplicates} exact duplicate row(s)", f"{round(duplicates / max(n_rows, 1) * 100, 2)}% of rows repeat exactly."))
    skewed = sorted([c for c in numeric if abs(c["stats"].get("skew") or 0) > 1], key=lambda c: -abs(c["stats"]["skew"]))
    if skewed:
        names = ", ".join(f"'{c['name']}' ({round(c['stats']['skew'], 2)})" for c in skewed[:3])
        ins.append(_insight("skew", "info", f"{len(skewed)} skewed numeric column(s)",
                            f"Long tails in {names}. Medians describe them better than means; log scales help charts and linear models.",
                            column=skewed[0]["name"]))
    extreme = sorted([c for c in numeric if (c.get("box") or {}).get("extreme")], key=lambda c: -c["box"]["extreme"])
    if extreme:
        c = extreme[0]
        ins.append(_insight("outliers", "warn", f"Extreme values in {len(extreme)} column(s)",
                            f"'{c['name']}' has {c['box']['extreme']} value(s) beyond 3×IQR (range {c['box']['min']:.4g} to {c['box']['max']:.4g}).",
                            column=c["name"]))
    if pairs:
        p = pairs[0]
        if abs(p["r"]) >= 0.5:
            ins.append(_insight("correlation", "warn" if abs(p["r"]) >= 0.9 else "info",
                                f"'{p['a']}' and '{p['b']}' move together (r = {p['r']})",
                                ("Nearly redundant: keep one for linear models. " if abs(p["r"]) >= 0.9 else "")
                                + f"{sum(1 for q in pairs if abs(q['r']) >= 0.5)} pair(s) have |r| ≥ 0.5."))
        else:
            ins.append(_insight("correlation", "info", "Numeric columns are largely independent",
                                f"The strongest pair is '{p['a']}' × '{p['b']}' at r = {p['r']}."))
    dominant = [c for c in columns if c.get("top_share", 0) >= 0.9 and c["kind"] != "numeric" and not c["is_target"]]
    if dominant:
        ins.append(_insight("dominant", "warn", f"{len(dominant)} near-constant column(s)",
                            ", ".join(f"'{c['name']}' ({round(c['top_share'] * 100)}% one value)" for c in dominant[:3]) + "."))
    if target_info:
        if problem_type == "classification" and target_info.get("classes"):
            cl = target_info["classes"]
            total = sum(c["count"] for c in cl) or 1
            share = cl[0]["count"] / total
            ins.append(_insight("target", "warn" if share > 0.75 else "good",
                                f"Target '{target}': {len(cl)} class(es), largest {round(share * 100, 1)}%",
                                "Imbalanced: judge models on balanced accuracy / F1, not accuracy." if share > 0.75
                                else "Classes are reasonably balanced."))
        elif target_info.get("box"):
            b = target_info["box"]
            ins.append(_insight("target", "info", f"Target '{target}' ranges {b['min']:.4g} to {b['max']:.4g}",
                                f"Median {b['median']:.4g}, middle half between {b['q1']:.4g} and {b['q3']:.4g}."))
        drv = target_info.get("drivers") or []
        if drv and drv[0]["association"] >= 0.1:
            ins.append(_insight("drivers", "good", f"'{drv[0]['feature']}' is the strongest driver of '{target}'",
                                "Top associations: " + ", ".join(f"{d['feature']} ({d['association']})" for d in drv[:4]) + "."))
        elif drv:
            ins.append(_insight("drivers", "bad", f"Nothing clearly drives '{target}'",
                                f"The strongest association is {drv[0]['association']} ('{drv[0]['feature']}'): too weak to predict from."))
    if diagnostics:
        hs = diagnostics.get("health_score") or {}
        drivers = hs.get("top_negative_drivers") or []
        score = hs.get("composite_score")
        if score is not None:
            ins.append(_insight("health", "good" if score >= 85 else "warn" if score >= 65 else "bad",
                                f"Health score {score}/100 ({hs.get('grade', '')})",
                                (drivers[0] + "." if drivers else hs.get("summary_text", ""))))
    if execution:
        ins.append(_insight("cleaning", "good" if execution.get("health_score_delta", 0) >= 0 else "warn",
                            f"Cleaning: {execution.get('before_health_score')} → {execution.get('after_health_score')}",
                            f"{len(execution.get('transformations_applied') or [])} step(s) produced {execution.get('cleaned_rows')} rows × "
                            f"{execution.get('cleaned_cols')} model-ready columns."))
    if leaderboard and leaderboard.get("models"):
        ev = leaderboard.get("evaluation") or {}
        best = leaderboard["models"][0]
        verdict = ev.get("verdict")
        tone = {"useful": "good", "weak": "warn", "no_signal": "bad"}.get(verdict, "info")
        ins.append(_insight("models", tone, f"Best: {best['model_name']} ({leaderboard['primary_metric']} {best['metric_display']})",
                            (leaderboard.get("best_model_summary") or "").replace("**", "")[:400]))

    return {
        "rows": n_rows,
        "cols": n_cols,
        "kinds": kinds,
        "target": target,
        "problem_type": problem_type,
        "columns": columns,
        "missing": missing,
        "duplicates": duplicates,
        "correlation": {"pairs": pairs[:12], "heatmap": heat},
        "target_info": target_info,
        "projection": projection,
        "insights": ins,
        "health": (diagnostics or {}).get("health_score"),
        "issues": [{"severity": i["severity"], "title": i["title"], "type": i["issue_type"]} for i in (diagnostics or {}).get("issues", [])],
        "execution": {k: execution.get(k) for k in ("before_health_score", "after_health_score", "health_score_delta", "cleaned_rows",
                                                    "cleaned_cols", "transformations_applied", "quality_audit")} if execution else None,
        "leaderboard": leaderboard,
        "report": report,
    }
