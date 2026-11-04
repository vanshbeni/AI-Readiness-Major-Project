"""
Live smoke test of the agents against the real OpenRouter key in backend/.env.

    cd backend
    venv\\Scripts\\python tests\\agents_live.py

Starts one backend on :8004 with a throwaway SQLite DB, runs autopilot on the titanic demo
(auto-approving every gate) and asks the analyst one question. Uses real (free) model calls.
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents_e2e import Client, start_backend  # noqa: E402
from app.core.config import settings  # noqa: E402

PORT = 8004


def main():
    if not settings.OPENROUTER_API_KEY.strip():
        sys.exit("OPENROUTER_API_KEY is not set in backend/.env")
    proc = start_backend(PORT, {"GEMINI_API_KEY": ""}, os.path.join(tempfile.mkdtemp(), "live.db"))
    ok = True
    try:
        c = Client(PORT)
        print("status:", c.call("GET", "/agents/status"))
        ds = c.call("POST", "/datasets/demo/titanic")
        c.call("POST", f"/datasets/{ds['id']}/objective", json={"problem_type": "classification", "target_column": "Survived"})

        t0 = time.time()
        run_id = c.call("POST", f"/datasets/{ds['id']}/agents/autopilot")["id"]
        while True:
            run = c.wait(run_id, timeout=900)
            if run["status"] != "waiting":
                break
            print(f"  gate: {run['gate'].get('kind') or run['gate'].get('title') or 'approval'} -> approving")
            c.call("POST", f"/agent-runs/{run_id}/resume", json={"approvals": {}, "answers": {}})
        print(f"\nautopilot: {run['status']} in {time.time() - t0:.0f}s, usage={run['usage']}")
        if run.get("error"):
            print("  error:", run["error"])
        ok &= run["status"] == "completed"
        for s in run["steps"]:
            if s["kind"] in ("critic", "fallback", "error"):
                print(f"  [{s['kind']}] {s['text'][:160]}")
        res = run.get("result") or {}
        print("  best model:", res.get("best_model"))
        report = res.get("report") or {}
        print(f"  report source: {report.get('source')}\n")
        print((report.get("markdown") or "")[:1500])

        t0 = time.time()
        question = "Which passenger class had the highest survival rate, and by how much?"
        run = c.wait(c.call("POST", f"/datasets/{ds['id']}/agents/ask", json={"question": question})["id"], timeout=600)
        res = run.get("result") or {}
        print(f"\nask: {run['status']} in {time.time() - t0:.0f}s, usage={run['usage']}")
        print("  Q:", question)
        print("  A:", res.get("answer"))
        print("  verification:", res.get("verification"))
        ok &= run["status"] == "completed"
    finally:
        proc.kill()
    print("\nLIVE OK" if ok else "\nLIVE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
