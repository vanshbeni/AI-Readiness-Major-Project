"""
End-to-end test of the unsupervised tasks (clustering, anomaly detection) and the summary page API, over HTTP,
rules-only (no LLM).

    cd backend
    venv\\Scripts\\python tests\\unsupervised_e2e.py
"""
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents_e2e import Client, start_backend  # noqa: E402
from cleaning_quality import messy_hr  # noqa: E402

PORT = 8008
FAILS = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def blobs():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(np.vstack([rng.normal(c, 0.6, (150, 3)) for c in ([0, 0, 0], [5, 5, 0], [0, 6, 4])]), columns=["a", "b", "c"])
    df["customer_id"] = [f"C{i:05d}" for i in range(len(df))]
    df.loc[rng.choice(len(df), 20, replace=False), "b"] = np.nan
    return df


def planted():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(0, 1, (500, 4)), columns=["p", "q", "r", "s"])
    df.iloc[:12] = rng.normal(0, 1, (12, 4)) * 8
    df["region"] = rng.choice(["north", "south", "east"], 500)
    return df


def run_task(c: Client, name: str, df: pd.DataFrame, task: str):
    print(f"\n[{name}] {task}")
    ds = c.call("POST", "/datasets/upload", files={"file": (f"{name}.csv", df.to_csv(index=False).encode())})
    obj = c.call("POST", f"/datasets/{ds['id']}/objective", json={"problem_type": task})
    check(obj["problem_type"] == task and obj["target_column"] == "", "objective saved without a target")
    diag = c.call("POST", f"/datasets/{ds['id']}/diagnose")
    check(diag["health_score"]["composite_score"] > 0, f"diagnosed, score {diag['health_score']['composite_score']}")
    check(not any(i["issue_type"] in ("class_imbalance", "target_leakage", "weak_target_signal") for i in diag["issues"]),
          "no target-only issues raised")
    ex = c.call("POST", f"/datasets/{ds['id']}/execute")
    check(ex["cleaned_rows"] > 0, f"cleaned {ex['cleaned_rows']}x{ex['cleaned_cols']}")
    lb = c.call("POST", f"/datasets/{ds['id']}/benchmark")
    ev = lb["evaluation"]
    check(lb["problem_type"] == task and ev.get("has_assignments"), "benchmark produced assignments")
    labelled = requests.get(f"{c.base}/datasets/{ds['id']}/export/assignments", headers=c.h)
    check(labelled.ok, f"labelled export ({labelled.status_code})")
    out = pd.read_csv(io.BytesIO(labelled.content)) if labelled.ok else pd.DataFrame()
    summ = c.call("GET", f"/datasets/{ds['id']}/summary?source=cleaned")
    check(summ["leaderboard"] and summ["insights"], "summary includes leaderboard and insights")
    check(summ["projection"] is None or summ["projection"]["color"] is not None, "summary projection coloured by the benchmark labels")
    return ds, lb, ev, out


def main():
    proc = start_backend(PORT, {"OPENROUTER_API_KEY": ""}, os.path.join(tempfile.mkdtemp(), "u.db"))
    try:
        c = Client(PORT)
        _, lb, ev, out = run_task(c, "blobs", blobs(), "clustering")
        check(ev["verdict"] == "useful", f"real groups found (verdict {ev['verdict']}, lift {ev['lift_over_baseline']})")
        check(ev["k"] == 3, f"three clusters found (k={ev['k']})")
        check("cluster" in out.columns and out["cluster"].nunique() >= 3, "every row has a cluster id")
        check(len(ev["clusters"]) >= 3 and all(p["traits"] for p in ev["clusters"] if not p["noise"]), "each cluster has distinguishing traits")
        check(any(m["is_baseline"] for m in lb["models"]), "shuffled-columns reference shown")
        check("customer_id" not in (ev.get("features") or []), "identifier not used as a clustering feature")

        _, lb, ev, _ = run_task(c, "messy_hr", messy_hr(), "clustering")
        check(ev["verdict"] == "no_signal", f"random HR data has no groups (verdict {ev['verdict']})")
        check(not any(m["is_recommended"] for m in lb["models"]), "nothing recommended when there is no structure")

        _, lb, ev, out = run_task(c, "planted", planted(), "anomaly_detection")
        check(ev["verdict"] == "useful", f"planted anomalies agreed on (verdict {ev['verdict']})")
        flagged = set(out.index[out["is_anomaly"] == 1]) if "is_anomaly" in out.columns else set()
        check(len(flagged & set(range(12))) >= 10, f"at least 10 of 12 planted rows flagged ({len(flagged & set(range(12)))})")
        check(ev["anomalies"] and ev["anomalies"][0]["reasons"], "top anomalies come with reasons")

        print("\n[autopilot + evaluate agent] clustering, rules only")
        ds = c.call("POST", "/datasets/upload", files={"file": ("blobs2.csv", blobs().to_csv(index=False).encode())})
        c.call("POST", f"/datasets/{ds['id']}/objective", json={"problem_type": "clustering", "target_column": ""})
        run_id = c.call("POST", f"/datasets/{ds['id']}/agents/autopilot")["id"]
        while True:
            run = c.wait(run_id)
            if run["status"] != "waiting":
                break
            c.call("POST", f"/agent-runs/{run_id}/resume", json={"approvals": {}, "answers": {}})
        check(run["status"] == "completed", f"autopilot completed for clustering ({run.get('error')})")
        ev_run = c.wait(c.call("POST", f"/datasets/{ds['id']}/agents/evaluate")["id"], until=("completed", "failed"))
        check(ev_run["status"] == "completed" and ev_run["result"]["saved"] is False, "evaluate agent explains there is no plan to choose")

        print("\n[supervised summary] titanic demo")
        ds = c.call("POST", "/datasets/demo/titanic")
        c.call("POST", f"/datasets/{ds['id']}/objective", json={"problem_type": "classification", "target_column": "Survived"})
        c.call("POST", f"/datasets/{ds['id']}/diagnose")
        s = c.call("GET", f"/datasets/{ds['id']}/summary?source=raw")
        drivers = [d["feature"] for d in (s["target_info"] or {}).get("drivers", [])]
        check(drivers and drivers[0] == "Sex", f"Sex is the top driver ({drivers[:3]})")
        check(s["target_info"]["drivers"][0].get("breakdown"), "top driver has a per-group breakdown")
        check(any(col.get("hist") for col in s["columns"]), "numeric histograms present")
        check(s["health"] and s["issues"], "health score and issues present")
        bad = requests.get(f"{c.base}/datasets/{ds['id']}/summary?source=cleaned", headers=c.h)
        check(bad.status_code == 404, "cleaned summary before cleaning is a clear 404")
    finally:
        proc.kill()
    print(f"\n{'ALL PASSED' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    for f in FAILS:
        print("  -", f)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
