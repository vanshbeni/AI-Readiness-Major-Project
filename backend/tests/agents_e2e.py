"""
End-to-end test of the agentic layer over HTTP, with a fake OpenRouter server (no real key needed).

    cd backend
    venv\\Scripts\\python tests\\agents_e2e.py            # fake LLM mode + rules-only mode

It starts: a fake OpenAI-compatible LLM on :8099, and two backend instances on a throwaway SQLite DB:
:8002 (OPENROUTER_BASE_URL -> fake LLM) and :8003 (no key -> deterministic rules only).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import requests

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
FAKE_PORT, LLM_PORT, RULES_PORT = 8099, 8002, 8003
CALLS = {"count": 0, "agents": {}}


# ─── Fake OpenRouter ─────────────────────────────────────────────────────────

def fake_answer(messages):
    system = messages[0]["content"]
    convo = " ".join(m["content"] for m in messages[1:])
    if "Understand agent" in system:
        payload = json.loads(messages[-1]["content"])
        cols = [s["column"] for s in payload["dataset"]["column_signals"]]
        cand = payload["candidates"]
        out = {"decisions": [], "rejected": [], "questions": [], "column_notes": {c: f"meaning of {c}" for c in cols[:3]}}
        if cand:
            first = cand[0]
            out["decisions"].append({**{k: first[k] for k in ("column", "action", "params")}, "confidence": 0.97, "reason": "Agent confirms."})
        out["decisions"].append({"column": "does_not_exist", "action": "ignore_column", "params": {}, "confidence": 0.9, "reason": "bogus"})
        return "agent:understand", "Here you go:\n```json\n" + json.dumps(out) + "\n```"
    if "Clean agent" in system:
        payload = json.loads(messages[-1]["content"])
        decisions = []
        for r in payload["recommendations"]:
            if r["issue_type"] == "missing_values" and r["current_method"] == "impute_mean":
                decisions.append({"rec_id": r["rec_id"], "approve": True, "method_key": "impute_median", "params": {}, "reason": "robust"})
            if r["issue_type"] == "outliers":
                pass
        decisions.append({"rec_id": "nope", "approve": True, "method_key": "x", "params": {}, "reason": "bogus"})
        return "agent:clean", json.dumps({"decisions": decisions})
    if "Evaluate agent" in system:
        payload = json.loads(messages[-1]["content"])
        metric = "balanced_accuracy" if payload["facts"]["problem_type"] == "classification" else "rmse"
        return "agent:evaluate", json.dumps({"metric": metric, "cv": "kfold", "time_column": None, "max_rows": 3000,
                                             "reason": "Balanced view of all classes."})
    if "Report agent" in system:
        if '"feedback"' in messages[-1]["content"]:
            return "agent:report", json.dumps({"markdown": "### Summary\n- Rows: {{F1}}, columns: {{F2}}.\n### Next steps\n- Export."})
        return "agent:report", json.dumps({"markdown": "### Summary\n- The file has {{F1}} rows and 999 typos."})
    if "Analyst agent" in system:
        turns = sum(1 for m in messages if m["role"] == "assistant")
        obs = [m["content"] for m in messages if m["role"] == "user" and m["content"].startswith("Observation:")]
        if turns == 0:
            return "agent:ask", json.dumps({"thought": "look at columns", "action": "schema", "args": {}})
        if turns == 1:
            return "agent:ask", json.dumps({"thought": "rate by group", "action": "sql", "args": {
                "query": 'SELECT "Sex", AVG("Survived") AS rate, COUNT(*) AS n FROM data GROUP BY 1 ORDER BY 2 DESC'}})
        if turns == 2:
            return "agent:ask", json.dumps({"thought": "chart it", "action": "chart", "args": {"table": "T1", "type": "bar", "x": "Sex", "y": "rate", "title": "Survival by sex"}})
        table = next((json.loads(o[len("Observation: "):]) for o in obs if '"table_id"' in o), None)
        if "Critic:" in convo and turns >= 4:
            row = table["rows"][0]
            return "agent:ask", json.dumps({"action": "final", "args": {"answer": f"{row[0]} has the highest survival rate, {round(row[1], 3)} [T1].", "charts": ["C1"]}})
        return "agent:ask", json.dumps({"action": "final", "args": {"answer": "Survival is 87.5% for one group [T1].", "charts": ["C1"]}})
    if "Senior Data Scientist explaining" in system:
        payload = json.loads(messages[-1]["content"].split("Data Payload:\n", 1)[1])
        return "explainer", json.dumps({"items": [{"index": p["index"], "explanation": f"Because of the stats for {p['column']}."} for p in payload]})
    return "unknown", "{}"


class FakeLLM(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.headers.get("Authorization") == "Bearer test-key"
        who, text = fake_answer(body["messages"])
        CALLS["count"] += 1
        CALLS["agents"][who] = CALLS["agents"].get(who, 0) + 1
        out = {"model": body["model"], "choices": [{"message": {"role": "assistant", "content": text}}],
               "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def start_backend(port, env_extra, db_path):
    try:
        requests.get(f"http://localhost:{port}/api/v1/health", timeout=1)
        raise RuntimeError(f"Port {port} is already in use (a previous test server?). Stop it and re-run.")
    except requests.RequestException:
        pass
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_path}", "GEMINI_API_KEY": "", **env_extra}
    proc = subprocess.Popen([PY, "-m", "uvicorn", "app.main:app", "--port", str(port)], cwd=BACKEND, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(90):
        try:
            if requests.get(f"http://localhost:{port}/api/v1/health", timeout=1).ok:
                return proc
        except requests.RequestException:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"backend on {port} did not start")


class Client:
    def __init__(self, port):
        self.base = f"http://localhost:{port}/api/v1"
        self.h = {"X-Client-Id": str(uuid.uuid4())}

    def call(self, method, path, **kw):
        r = requests.request(method, self.base + path, headers=self.h, timeout=300, **kw)
        if not r.ok:
            raise AssertionError(f"{method} {path} -> {r.status_code}: {r.text[:400]}")
        return r.json() if r.content else None

    def wait(self, run_id, until=("waiting", "completed", "failed", "cancelled"), timeout=240):
        t0 = time.time()
        while time.time() - t0 < timeout:
            run = self.call("GET", f"/agent-runs/{run_id}")
            if run["status"] in until:
                return run
            time.sleep(0.5)
        raise AssertionError(f"run {run_id} timed out in status {run['status']}")


def messy_dataset(n=240):
    rng = np.random.default_rng(5)
    hours = rng.integers(1, 80, n).astype(object)
    hours[rng.choice(n, 25, replace=False)] = "?"
    y = rng.choice(["yes", "no"], n, p=[0.4, 0.6])
    return pd.DataFrame({
        "customer": [f"user{i}@mail.com" for i in range(n)],
        "hours": hours,
        "revenue": [f"${v:,}" for v in rng.integers(1000, 90000, n)],
        "city": rng.choice(["Delhi", "delhi ", "DELHI", "Mumbai", "mumbai"], n),
        "visit": pd.date_range("2024-01-01", periods=n).strftime("%d/%m/%Y"),
        "leak": np.where(y == "yes", 1, 0),
        "score": rng.normal(50, 10, n).round(2),
        "churned": y,
    })


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


FAILS = []


def flow_llm(c: Client):
    print("\n[LLM mode] upload messy dataset + understand agent")
    csv = messy_dataset().to_csv(index=False).encode()
    ds = c.call("POST", "/datasets/upload", files={"file": ("messy.csv", csv)})
    c.call("POST", f"/datasets/{ds['id']}/objective", json={"problem_type": "classification", "target_column": "churned"})
    status = c.call("GET", "/agents/status")
    check(status["configured"] is True, "agents/status reports configured")

    run = c.wait(c.call("POST", f"/datasets/{ds['id']}/agents/understand")["id"])
    check(run["status"] == "waiting", f"understand run waits at gate (status={run['status']})")
    items = run["gate"]["items"]
    actions = {(i["column"], i["action"]) for i in items}
    for expect in [("visit", "parse_dayfirst"), ("city", "normalize_categories"), ("customer", "ignore_column")]:
        check(expect in actions, f"proposal {expect}")
    # '?' placeholders and '$12,345' strings are normalised automatically before any agent runs
    check(not ({("hours", "treat_as_missing"), ("revenue", "parse_numeric")} & actions), "auto-normalised columns not re-proposed")
    check(not any(i["column"] == "does_not_exist" for i in items), "critic dropped the bogus agent decision")
    check(any(i["source"].startswith("rules + agent") for i in items), "agent confirmation merged into a rule candidate")
    check(any(s["kind"] == "critic" for s in run["steps"]), "critic step logged")
    approvals = {i["id"]: True for i in items}
    c.call("POST", f"/agent-runs/{run['id']}/resume", json={"approvals": approvals, "answers": {}})
    run = c.wait(run["id"], until=("completed", "failed"))
    check(run["status"] == "completed" and len(run["result"]["applied"]) >= 3, f"understand applied decisions ({run['result']})")

    diag = c.call("POST", f"/datasets/{ds['id']}/diagnose")
    found = {(i["issue_type"], i["column"]) for i in diag["issues"]}
    check(("disguised_missing", "hours") in found, "hidden '?' placeholders in hours reported")
    check(("numeric_stored_as_text", "revenue") in found, "revenue reported as numbers stored as text")
    types = {col["name"]: col["inferred_type"] for col in diag["profile"]["columns"]}
    check(types.get("revenue") == "numeric", f"revenue now numeric ({types.get('revenue')})")
    rev = next(col for col in diag["profile"]["columns"] if col["name"] == "revenue")["numeric_stats"]
    check(rev["min"] >= 1000, f"'$12,345' parsed as 12345, not 12.345 (min={rev['min']})")
    check(types.get("hours") == "numeric", f"hours now numeric ({types.get('hours')})")
    check("customer" not in types, "PII column excluded")
    check(any(i["issue_type"] == "target_leakage" and i["column"] == "leak" for i in diag["issues"]), "leakage issue detected")
    check(any(i["issue_type"] == "unparsed_datetime_feature" and i["column"] == "visit" for i in diag["issues"]), "visit date will be parsed")
    check(any(r["explanation_source"] == "ai" for r in diag["recommendations"]), "explanations came from the (fake) LLM")
    city_top = next(col for col in diag["profile"]["columns"] if col["name"] == "city")["categorical_stats"]["top_categories"]
    check(len(city_top) == 2, f"city variants merged ({[t['value'] for t in city_top]})")
    check({t["value"] for t in city_top} == {"Delhi", "Mumbai"}, "canonical spellings are the clean forms")

    print("[LLM mode] clean agent")
    run = c.wait(c.call("POST", f"/datasets/{ds['id']}/agents/clean")["id"])
    check(run["status"] in ("waiting", "completed"), f"clean run status {run['status']}")
    if run["status"] == "waiting":
        c.call("POST", f"/agent-runs/{run['id']}/resume", json={"approvals": {}, "answers": {}})
        run = c.wait(run["id"], until=("completed", "failed"))
    check(run["status"] == "completed", "clean run completed")

    print("[LLM mode] execute + evaluate + benchmark + report")
    ex = c.call("POST", f"/datasets/{ds['id']}/execute")
    steps = " | ".join(ex["transformations_applied"])
    check("Prep (auto): treated" in steps and "Prep (auto): read 'revenue' as numbers" in steps, "prep steps executed first")
    check("target leakage" in steps, "leaking column dropped")
    check(any(s.startswith("Datetime Engineering: Parsed 'visit'") for s in ex["transformations_applied"]), "visit parsed into date parts")
    script = requests.get(f"{c.base}/datasets/{ds['id']}/export/pipeline-script", headers=c.h).text
    check("replace_with_nan(df" in script and "parse_dayfirst(df" in script, "pipeline script reproduces the prep")

    run = c.wait(c.call("POST", f"/datasets/{ds['id']}/agents/evaluate")["id"])
    check(run["status"] == "waiting" and run["gate"]["plan"]["metric"] == "balanced_accuracy", "evaluate proposes agent plan")
    c.call("POST", f"/agent-runs/{run['id']}/resume", json={"approvals": {"plan": True}})
    c.wait(run["id"], until=("completed",))
    lb = c.call("POST", f"/datasets/{ds['id']}/benchmark")
    check(lb["evaluation"]["metric"] == "balanced_accuracy" and lb["evaluation"]["source"] == "agent", "benchmark used the agent plan")
    check(any(m["is_baseline"] for m in lb["models"]) and lb["models"][0]["metric_std"] is not None, "baseline + std reported")

    run = c.wait(c.call("POST", f"/datasets/{ds['id']}/agents/report")["id"])
    rep = run["result"]["report"]
    check(rep["source"] == "agent" and "999" not in rep["markdown"], "critic rejected typed number; revised report accepted")
    pol = c.call("GET", f"/datasets/{ds['id']}/agents/policy")
    check(pol["report"] and pol["eval_plan"] and pol["column_overrides"], "policy stores overrides, plan and report")


def flow_autopilot_and_ask(c: Client):
    print("\n[LLM mode] autopilot on titanic demo")
    ds = c.call("POST", "/datasets/demo/titanic")
    c.call("POST", f"/datasets/{ds['id']}/objective", json={"problem_type": "classification", "target_column": "Survived"})
    run_id = c.call("POST", f"/datasets/{ds['id']}/agents/autopilot")["id"]
    gates = 0
    while True:
        run = c.wait(run_id)
        if run["status"] != "waiting":
            break
        gates += 1
        c.call("POST", f"/agent-runs/{run_id}/resume", json={"approvals": {}, "answers": {}})
        time.sleep(0.3)
    check(run["status"] == "completed", f"autopilot completed (status={run['status']}, error={run.get('error')})")
    check(run["result"] and run["result"].get("best_model"), f"autopilot result has best model ({(run['result'] or {}).get('best_model')})")
    check(run["usage"]["calls"] >= 4, f"usage tracked ({run['usage']})")
    session = c.call("GET", f"/datasets/{ds['id']}/session")
    check(session["leaderboard"] is not None and session["execution"] is not None, "session restored with execution + leaderboard")
    print(f"  info: {gates} gate(s), {run['step_count']} trace steps")

    print("[LLM mode] ask the analyst")
    run = c.wait(c.call("POST", f"/datasets/{ds['id']}/agents/ask", json={"question": "Which sex survived more?"})["id"])
    res = run["result"] or {}
    check(run["status"] == "completed", f"ask completed ({run.get('error')})")
    check("[T1]" in res.get("answer", ""), f"answer cites T1: {res.get('answer')}")
    check(any(s["kind"] == "critic" and "not backed" in s["text"] for s in run["steps"]), "critic caught unsupported 87.5% and forced revision")
    check(res.get("verification", {}).get("unverified_numbers") == [], f"final answer fully verified ({res.get('verification')})")
    check(len(res.get("charts", [])) == 1 and res["charts"][0]["rows"], "chart spec returned with data")
    bad = c.call("POST", f"/datasets/{ds['id']}/agents/ask", json={"question": "x"})
    check(bad["status"] in ("queued", "running", "completed"), "second ask accepted")


def flow_rules(c: Client):
    print("\n[rules mode] no LLM key")
    csv = messy_dataset().to_csv(index=False).encode()
    ds = c.call("POST", "/datasets/upload", files={"file": ("messy.csv", csv)})
    c.call("POST", f"/datasets/{ds['id']}/objective", json={"problem_type": "classification", "target_column": "churned"})
    check(c.call("GET", "/agents/status")["configured"] is False, "status reports not configured")
    run = c.wait(c.call("POST", f"/datasets/{ds['id']}/agents/autopilot")["id"])
    check(run["status"] == "waiting" and len(run["gate"]["items"]) >= 3, "rules-only understand still proposes decisions")
    run_id = run["id"]
    while run["status"] == "waiting":
        c.call("POST", f"/agent-runs/{run_id}/resume", json={"approvals": {}, "answers": {}})
        run = c.wait(run_id)
    check(run["status"] == "completed", f"rules-only autopilot completed ({run.get('error')})")
    check(run["result"]["report"]["source"] == "rules", "deterministic report used without LLM")
    ask = c.wait(c.call("POST", f"/datasets/{ds['id']}/agents/ask", json={"question": "hi"})["id"])
    check("OPENROUTER_API_KEY" in ask["result"]["answer"], "ask explains that an LLM key is needed")
    run = c.call("POST", f"/datasets/{ds['id']}/agents/understand")
    c.call("POST", f"/agent-runs/{run['id']}/cancel")
    check(c.call("GET", f"/agent-runs/{run['id']}")["status"] in ("cancelled", "completed"), "cancel works")


def main():
    server = ThreadingHTTPServer(("127.0.0.1", FAKE_PORT), FakeLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    tmp = tempfile.mkdtemp()
    procs = []
    try:
        procs.append(start_backend(LLM_PORT, {"OPENROUTER_API_KEY": "test-key",
                                              "OPENROUTER_BASE_URL": f"http://127.0.0.1:{FAKE_PORT}/v1"}, os.path.join(tmp, "a.db")))
        procs.append(start_backend(RULES_PORT, {"OPENROUTER_API_KEY": ""}, os.path.join(tmp, "b.db")))
        flow_llm(Client(LLM_PORT))
        flow_autopilot_and_ask(Client(LLM_PORT))
        flow_rules(Client(RULES_PORT))
    finally:
        for p in procs:
            p.kill()
        server.shutdown()
    print(f"\nfake LLM calls: {CALLS}")
    print(f"\n{'ALL PASSED' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
