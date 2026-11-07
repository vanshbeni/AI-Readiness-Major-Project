"""
Output-quality battery: runs many kinds of datasets through the full pipeline (Understand prep -> diagnose -> clean
-> execute) and reviews every exported CSV like an independent data scientist would. Two modes:
  rules      - no LLM key, deterministic agents
  adversary  - a fake LLM that deliberately proposes bad choices (keep top-N of name/code columns, mean imputation,
               parsing codes as numbers, capping everything); the critics and the output audit must neutralise them.

    cd backend
    venv\\Scripts\\python tests\\cleaning_quality.py
"""
import io
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents_e2e import Client, messy_dataset, start_backend  # noqa: E402

FAKE_PORT, ADV_PORT, RULES_PORT = 8098, 8006, 8007
CODE_RE = r"^[A-Za-z]+[\s\-_/]*\d"
FAILS = []


# ─── Datasets ────────────────────────────────────────────────────────────────

def titanic_real_like(n=400, seed=1):
    rng = np.random.default_rng(seed)
    titles = rng.choice(["Mr", "Mrs", "Miss", "Master", "Dr", "Rev"], n, p=[0.55, 0.15, 0.2, 0.06, 0.02, 0.02])
    first = rng.choice(["John", "William", "Mary", "Anna", "James", "Elizabeth", "Thomas", "Karl", "Ali", "Juan"], n)
    last = [f"Surname{rng.integers(0, 10_000)}" for _ in range(n)]
    pclass = rng.choice([1, 2, 3], n, p=[0.25, 0.2, 0.55])
    age = np.where(titles == "Master", rng.uniform(1, 12, n), rng.normal(32, 12, n).clip(14, 80)).round(1)
    age[rng.choice(n, 80, replace=False)] = np.nan
    fare = np.round(rng.lognormal(2.6, 0.9, n) * np.where(pclass == 1, 4, 1), 2)
    cabin = np.where((pclass == 1) | (rng.random(n) < 0.08),
                     [f"{rng.choice(list('ABCDE'))}{rng.integers(1, 130)}" for _ in range(n)], None)
    ticket = [rng.choice(["PC ", "A/5 ", "STON/O2. ", "", "", ""]) + str(rng.integers(1000, 400000)) for _ in range(n)]
    surv = (rng.random(n) < (0.2 + 0.5 * np.isin(titles, ["Mrs", "Miss"]) + 0.15 * (pclass == 1))).astype(int)
    return pd.DataFrame({"PassengerId": np.arange(1, n + 1), "Survived": surv, "Pclass": pclass,
                         "Name": [f"{l}, {t}. {f}" for l, t, f in zip(last, titles, first)],
                         "Sex": np.where(np.isin(titles, ["Mrs", "Miss"]), "female", "male"), "Age": age,
                         "SibSp": rng.choice([0, 1, 2, 3, 5, 8], n, p=[0.68, 0.23, 0.04, 0.03, 0.01, 0.01]),
                         "Ticket": ticket, "Fare": fare, "Cabin": cabin,
                         "Embarked": rng.choice(["S", "C", "Q", None], n, p=[0.7, 0.19, 0.09, 0.02])})


def kitchen_sink(n=500, seed=2):
    """Many edge cases at once, regression target."""
    rng = np.random.default_rng(seed)
    city = rng.zipf(1.6, n).clip(1, 60)
    tags = [", ".join(rng.choice(["red", "blue", "green", "x", "y"], rng.integers(1, 3), replace=False)) for _ in range(n)]
    price = rng.lognormal(4, 0.6, n)
    df = pd.DataFrame({
        "row_number": np.arange(n),
        "sku": [f"{rng.choice(['ELEC', 'HOME', 'TOY', 'BOOK'])}-{rng.integers(10000, 99999)}" for _ in range(n)],
        "unique_note": [f"note {i} {rng.integers(0, 1e9)}" for i in range(n)],
        "city": [f"City{c}" for c in city],
        "mostly_empty": np.where(rng.random(n) < 0.93, np.nan, rng.normal(10, 2, n)),
        "rare_flag": np.isin(np.arange(n), [3, 77, 300]).astype(float),
        "constant": "same",
        "discount": [f"{v}%" for v in rng.integers(0, 40, n)],
        "list_price": [f"${v:,.2f}" for v in price * 1.2],
        "rating": np.where(rng.random(n) < 0.1, "N/A", rng.integers(1, 6, n).astype(str)),
        "in_stock": rng.choice(["yes", "no"], n, p=[0.8, 0.2]),
        "created": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "tags": tags,
        "weight_kg": np.where(rng.random(n) < 0.15, np.nan, rng.gamma(2, 2, n)),
        "ratio": np.where(rng.random(n) < 0.02, np.inf, rng.normal(1, 0.1, n)),
        "price": price.round(2),
    })
    df.loc[rng.choice(n, 4, replace=False), "price"] = rng.uniform(5000, 9000, 4)
    return df


def tiny_ids(n=60, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "uuid": [f"u-{i:04d}-{rng.integers(0, 9999)}" for i in range(n)],
        "group": rng.choice(["a", "b"], n),
        "score": rng.normal(0, 1, n).round(3),
        "label": rng.choice(["pos", "neg"], n),
    })


def messy_hr(n=1000, seed=4):
    """Same shape as a classic 'messy HR' export: every column random, numbers written as words, placeholder text,
    whitespace-only cells, five date formats, dummy emails / phones. Looks tidy, carries no signal."""
    rng = np.random.default_rng(seed)
    age = rng.choice(["25", "thirty", "35", "40", "50", None], n)
    salary = rng.choice(["50000", "55000", "SIXTY THOUSAND", "65000", "70000", " NAN "], n)
    return pd.DataFrame({
        "Name": rng.choice(["alice", "bob", "charlie", "david", "eve", "frank", "grace", "hannah", "ivy", "jack"], n),
        "Age": age, "Salary": salary,
        "Gender": rng.choice(["Male", "Female", "Other"], n),
        "Department": rng.choice(["HR", "Finance", "IT", "Sales", "Marketing"], n),
        "Position": rng.choice(["Manager", "Director", "Clerk", "Analyst", "Assistant"], n),
        "Joining Date": rng.choice(["April 5, 2018", "2020/02/20", "01/15/2020", "03-25-2019", "2019.12.01"], n),
        "Performance Score": rng.choice(list("ABCDF"), n),
        "Email": rng.choice(["email@example.com", "user@domain.com", "name@company.org", None], n, p=[0.2, 0.2, 0.2, 0.4]),
        "Phone Number": rng.choice(["123-456-7890", "098-765-4321", "555-555-5555", "   ", None], n),
    })


# name, source, target, problem type, expectations (health score bounds / benchmark verdicts)
DATASETS = [
    ("titanic demo", "demo:titanic", "Survived", "classification", {"min_score": 70, "verdict": {"useful", "weak"}}),
    ("churn demo", "demo:churn", "Churn", "classification", {"min_score": 70}),
    ("housing demo", "demo:housing", "MedianHouseValue", "regression", {"min_score": 70, "verdict": {"useful"}}),
    ("titanic real-like", titanic_real_like, "Survived", "classification", {}),
    ("messy", messy_dataset, "churned", "classification", {}),
    ("kitchen sink", kitchen_sink, "price", "regression", {}),
    ("tiny ids", tiny_ids, "label", "classification", {}),
    ("messy hr", messy_hr, "Salary", "regression",
     {"max_score": 65, "verdict": {"no_signal"}, "dropped": ["Name", "Email", "Phone Number"], "numeric": ["Age", "Salary"]}),
    ("messy hr classification", messy_hr, "Performance Score", "classification", {"max_score": 65, "verdict": {"no_signal"}}),
]


# ─── Adversarial fake LLM ────────────────────────────────────────────────────

def adversary_answer(messages):
    system, last = messages[0]["content"], messages[-1]["content"]
    if "Understand agent" in system:
        payload = json.loads(last)
        decisions = [{"column": s["column"], "action": "parse_numeric", "params": {"decimal": "."}, "confidence": 0.99,
                      "reason": "looks numeric"} for s in payload["dataset"]["column_signals"] if s.get("samples")]
        return json.dumps({"decisions": decisions, "rejected": [], "questions": [], "column_notes": {}})
    if "Clean agent" in system:
        payload = json.loads(last)
        out = []
        for r in payload["recommendations"]:
            keys = [a["method_key"] for a in r["alternatives"]]
            if "group_rare" in keys:
                out.append({"rec_id": r["rec_id"], "approve": True, "method_key": "group_rare", "params": {"keep_top": 50}, "reason": "keep"})
            elif "impute_mean" in keys:
                out.append({"rec_id": r["rec_id"], "approve": True, "method_key": "impute_mean", "params": {}, "reason": "mean"})
            elif r["issue_type"] == "statistical_outliers":
                out.append({"rec_id": r["rec_id"], "approve": True, "method_key": "cap_outliers", "params": {}, "reason": "cap all"})
        return json.dumps({"decisions": out})
    if "Evaluate agent" in system:
        return json.dumps({"metric": "auto", "cv": "kfold", "time_column": None, "max_rows": 3000, "reason": "default"})
    if "Report agent" in system:
        return json.dumps({"markdown": "### Summary\n- Rows: {{F1}}."})
    if "Senior Data Scientist explaining" in system:
        payload = json.loads(last.split("Data Payload:\n", 1)[1])
        return json.dumps({"items": [{"index": p["index"], "explanation": "x"} for p in payload]})
    return "{}"


class Adversary(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = adversary_answer(body["messages"])
        data = json.dumps({"model": body["model"], "choices": [{"message": {"role": "assistant", "content": text}}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ─── Independent reviewer ────────────────────────────────────────────────────

def check(cond, name, msg):
    if not cond:
        FAILS.append(f"{name}: {msg}")
        print(f"    FAIL {msg}")


def review(name, raw: pd.DataFrame, out: pd.DataFrame, target: str, dictionary: list, script: str, prepared_raw: pd.DataFrame):
    n = len(out)
    support = max(5, int(np.ceil(0.01 * n)))
    feats = [c for c in out.columns if c != target]
    check(target in out.columns and out[target].notna().all(), name, "target present without blanks")
    num = out[feats].select_dtypes(include=[np.number])
    check(len(num.columns) == len(feats), name, f"all features numeric (text left: {[c for c in feats if c not in num.columns]})")
    check(not out[feats].isna().any().any(), name, f"no blanks (cols: {[c for c in feats if out[c].isna().any()]})")
    check(not np.isinf(num.to_numpy(dtype=float)).any(), name, "no infinite values")
    for c in num.columns:
        s = out[c]
        vals = set(pd.unique(s))
        if vals <= {0, 1}:
            minority = int(min((s == 1).sum(), (s == 0).sum()))
            check(minority >= support, name, f"indicator '{c}' covers {minority} rows (< {support}): memorises rows")
            check(pd.api.types.is_integer_dtype(s), name, f"0/1 column '{c}' stored as {s.dtype}")
        check(s.nunique() > 1, name, f"'{c}' is constant")
        if pd.api.types.is_integer_dtype(s) and s.is_unique and len(s) > 20:
            check(not np.all(np.diff(np.sort(s.to_numpy())) == 1), name, f"'{c}' is a running identifier")
    for c in raw.columns:
        if c == target or c not in out.columns or raw[c].dtype != object:
            continue
        codes = raw[c].dropna().astype(str).str.match(CODE_RE).mean()
        check(not (codes >= 0.8 and pd.api.types.is_numeric_dtype(out[c])), name, f"alphanumeric codes in '{c}' were turned into numbers")
    base = prepared_raw
    for c in feats:
        if c in base.columns and base[c].isna().mean() > 0.7:
            check(False, name, f"'{c}' was {round(base[c].isna().mean() * 100)}% blank but kept as filled values")
    for c in feats:
        if c not in base.columns or not pd.api.types.is_numeric_dtype(base[c]) or out[c].nunique() <= 10:
            continue
        before = base[c].dropna().value_counts()
        after = out[c].value_counts()
        top_val, top_after = after.index[0], int(after.iloc[0])
        top_before = int(before.get(top_val, 0))
        if top_after - top_before >= support:
            flags = [f for f in out.columns if f in (f"{c}_was_missing", f"{c}_was_capped")]
            check(bool(flags), name, f"'{c}' gained a spike of {top_after - top_before} identical values ({top_val}) with no flag marking them")
    described = {d["column"]: d for d in dictionary}
    check(set(described) == set(out.columns), name, "data dictionary covers every column")
    check(all(d.get("meaning") for d in dictionary), name, "every dictionary row has a meaning")
    ns = {}
    exec(compile(script, "pipeline.py", "exec"), ns)
    replay = ns["clean_data"](raw.copy())
    same = replay.shape == out.shape and list(replay.columns) == list(out.columns)
    if same:
        a = replay.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        b = out.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        same = np.allclose(a, b, equal_nan=True, atol=1e-6)
        if not same:
            bad = [c for i, c in enumerate(out.columns) if not np.allclose(a[:, i], b[:, i], equal_nan=True, atol=1e-6)]
            print(f"    differing columns: {bad[:5]}; replay {replay[bad[0]].head(3).tolist()} vs csv {out[bad[0]].head(3).tolist()}")
    else:
        print(f"    replay columns {list(replay.columns)[:12]} vs csv {list(out.columns)[:12]}")
    check(same, name, "exported script reproduces the exported CSV")


def run_mode(label, client: Client, adversarial: bool):
    print(f"\n=== {label} ===")
    only = os.environ.get("ONLY", "").lower()
    for name, src, target, ptype, expect in DATASETS:
        if only and not any(o.strip() and o.strip() in name for o in only.split(",")):
            continue
        print(f"  [{name}]")
        if isinstance(src, str):
            ds = client.call("POST", f"/datasets/demo/{src.split(':')[1]}")
        else:
            ds = client.call("POST", "/datasets/upload", files={"file": (f"{name}.csv", src().to_csv(index=False).encode())})
        client.call("POST", f"/datasets/{ds['id']}/objective", json={"problem_type": ptype, "target_column": target})
        run_id = client.call("POST", f"/datasets/{ds['id']}/agents/understand")["id"]
        run = client.wait(run_id)
        if run["status"] == "waiting":
            client.call("POST", f"/agent-runs/{run_id}/resume", json={"approvals": {}, "answers": {}})
            client.wait(run_id, until=("completed", "failed"))
        diag = client.call("POST", f"/datasets/{ds['id']}/diagnose")
        score = diag["health_score"]["composite_score"]
        if "min_score" in expect:
            check(score >= expect["min_score"], name, f"health score {score} below {expect['min_score']} for a usable dataset")
        if "max_score" in expect:
            check(score <= expect["max_score"], name, f"health score {score} above {expect['max_score']} for junk data")
        run_id = client.call("POST", f"/datasets/{ds['id']}/agents/clean")["id"]
        run = client.wait(run_id)
        critic = [s["text"] for s in run["steps"] if s["kind"] == "critic"]
        if run["status"] == "waiting":
            client.call("POST", f"/agent-runs/{run_id}/resume", json={"approvals": {}, "answers": {}})
            client.wait(run_id, until=("completed", "failed"))
        ex = client.call("POST", f"/datasets/{ds['id']}/execute")
        out = pd.read_csv(io.BytesIO(requests.get(f"{client.base}/datasets/{ds['id']}/export/cleaned-csv", headers=client.h).content))
        script = requests.get(f"{client.base}/datasets/{ds['id']}/export/pipeline-script", headers=client.h).text
        dict_csv = requests.get(f"{client.base}/datasets/{ds['id']}/export/data-dictionary", headers=client.h)
        check(dict_csv.ok, name, f"data dictionary export ({dict_csv.status_code})")
        raw = source_frame(src)
        prepared = prepared_frame(script, raw)
        review(name, raw, out, target, ex["data_dictionary"], script, prepared)
        fixed = [a["check"] for a in ex["quality_audit"] if a["status"] == "fixed"]
        still_open = [a["detail"] for a in ex["quality_audit"] if a["status"] == "open"]
        check(not still_open, name, f"audit left open defects: {still_open}")
        for c in expect.get("dropped", []):
            check(not any(col == c or col.startswith(f"{c}_") for col in out.columns), name, f"'{c}' should not be a feature")
        for c in expect.get("numeric", []):
            check(c in out.columns and pd.api.types.is_numeric_dtype(out[c]), name, f"'{c}' should be numeric")
        verdict = None
        if "verdict" in expect:
            lb = client.call("POST", f"/datasets/{ds['id']}/benchmark")
            verdict = lb["evaluation"].get("verdict")
            check(verdict in expect["verdict"], name, f"benchmark verdict {verdict}, expected {expect['verdict']}")
        print(f"    score={score}" + (f" verdict={verdict}" if verdict else ""))
        print(f"    {ex['cleaned_rows']}x{ex['cleaned_cols']}  audit fixed={fixed or 'none'}"
              + (f"  critic={len(critic)}" if adversarial else ""))
        print(f"    columns: {', '.join(out.columns)}")


def source_frame(src):
    if isinstance(src, str):
        from app.api.v1.endpoints.datasets import DEMO_DATASETS
        return DEMO_DATASETS[src.split(":")[1]]["generator"]()
    return pd.read_csv(io.BytesIO(src().to_csv(index=False).encode()))


def prepared_frame(script: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Raw data after the approved 'Prep:' steps only, i.e. what cleaning started from."""
    lines = script.splitlines()
    start = lines.index("def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:")
    body = [l for l in lines[start + 3:] if l.startswith("    df = ")]
    prep = [l for l in body if l.split("(")[0].strip().split("= ")[1] in ("replace_with_nan", "parse_numeric", "parse_dayfirst", "map_values")]
    ns = {}
    exec(compile("\n".join(lines[:start]), "t.py", "exec"), ns)
    df = raw.copy()
    df.columns = [str(c) for c in df.columns]
    for l in prep:
        df = eval(l.split("= ", 1)[1], ns, {"df": df})
    return df


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    server = ThreadingHTTPServer(("127.0.0.1", FAKE_PORT), Adversary)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    tmp = tempfile.mkdtemp()
    procs = []
    try:
        procs.append(start_backend(RULES_PORT, {"OPENROUTER_API_KEY": ""}, os.path.join(tmp, "r.db")))
        procs.append(start_backend(ADV_PORT, {"OPENROUTER_API_KEY": "test-key",
                                              "OPENROUTER_BASE_URL": f"http://127.0.0.1:{FAKE_PORT}/v1"}, os.path.join(tmp, "a.db")))
        run_mode("rules mode", Client(RULES_PORT), False)
        run_mode("adversarial LLM mode", Client(ADV_PORT), True)
    finally:
        for p in procs:
            p.kill()
        server.shutdown()
    print(f"\n{'ALL PASSED' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    for f in FAILS:
        print("  -", f)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
