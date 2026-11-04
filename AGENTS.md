# AGENTS.md: Nebulus (AI Data Readiness Platform)

Read this before making any change. It is the permanent project brief for AI coding agents.
Full detail lives in `ARCHITECTURE.md`; this file is the short version plus rules.

## What this project is

Nebulus is an explainable pre-ML data readiness tool. A user uploads a CSV/Excel file, picks a target column,
and the system profiles the data, scores it 0-100, recommends fixes with explanations, applies only the fixes the
user approves, re-scores, benchmarks ML models with leak-free cross-validation, and exports artifacts
(cleaned CSV, reproducible Python script, PDF report, JSON manifest).

## Stack

| Area | Tech |
|---|---|
| Backend | Python, FastAPI, Pydantic v2, SQLAlchemy 2 (SQLite default, PostgreSQL optional) |
| Data/ML | pandas, NumPy, scikit-learn, imbalanced-learn, optional XGBoost / LightGBM |
| Reports | ReportLab (PDF) |
| AI | Google Gemini REST API (optional; falls back to statistical explanations) |
| Frontend | Next.js 14 (App Router), React 18, TypeScript, plain CSS (no Tailwind, no animation lib) |
| Monorepo | npm workspaces + Turborepo (`frontend/`) |

## Repo map

```text
backend/app/
  main.py                      FastAPI app, CORS, lifespan (init_db + file cleanup)
  core/{config,database,storage}.py
  models/dataset.py            SQLAlchemy ORM
  schemas/dataset.py           Pydantic request/response models
  api/deps.py                  client id, ownership, snapshot, invalidate_execution
  api/v1/router.py
  api/v1/endpoints/            health, datasets, objectives, diagnostics,
                               recommendations, pipeline, benchmarks, exports
  engine/                      PURE logic, no web deps:
                               profiler, detector, scorer, rules, explainer,
                               transforms, executor, benchmark, reporter
frontend/apps/
  dashboard/  (port 3000)      5-step wizard; state lives in src/app/page.tsx
                               API gateway: src/services/api.ts
  landing/    (port 3001)      marketing page
```

## Commands

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # Windows: copy .env.example .env
uvicorn app.main:app --reload --port 8000           # docs: http://localhost:8000/docs

# Frontend
cd frontend
npm install
npm run dev              # dashboard :3000 + landing :3001
npm run dev:dashboard
npm run check-types      # tsc --noEmit across workspaces
npm run build
```

Do not run `next build` while `next dev` is running for the same app (shared `.next` folder breaks the dev server).

## Pipeline (the core flow)

`upload -> objective -> diagnose -> approve fixes -> execute -> benchmark -> export`

Engine data flow:
`profiler -> detector -> scorer(before) -> rules -> explainer -> [user approvals] -> executor -> re-profile -> scorer(after) -> benchmark`

## Invariants (do not break these)

1. **Rules decide, the LLM only explains.** Fix selection is deterministic in `engine/rules.py`. Gemini never chooses or changes a method.
2. **Human in the loop.** Nothing runs until the user approves and clicks Execute. Destructive fixes default to `is_approved=False`.
3. **`Recommendation.params` is the source of truth for execution.** It stores `stats_context` including `method_key` and fitted values (bounds, medians, category lists). The executor reads these instead of recomputing, so what the user approved is exactly what runs.
4. **`engine/transforms.py` must stay pure and self-contained.** It is embedded verbatim into the exported pipeline script via `inspect.getsource`. Only `re`, `warnings`, `numpy`, `pandas` imports. Every function must be deterministic given its arguments.
5. **No leakage in benchmarking.** Imputation, scaling and SMOTE are fitted inside each CV fold (sklearn/imblearn Pipeline). Never move them outside.
6. **No scaling, SMOTE or blanket fills in the executor.** Those belong in model training.
7. **`df_cleaned` (not encoded) is used for the after-score; `df_model_ready` (encoded) is exported and benchmarked.** Keep this split.
8. **Invalidation chain.** Changing the objective clears diagnostics and execution. Changing approvals clears execution (jobs, cleaned CSV, after-score, benchmarks, snapshot). Use `invalidate_execution()` in `api/deps.py`; do not hand-roll it.
9. **Ownership.** Every `/datasets/*` route uses `get_owned_dataset`. Foreign or unknown datasets return 404 (no existence leak). Raw file missing returns 410.
10. **Concurrency.** Heavy endpoints (diagnose, execute, benchmark, explain) use `dataset_lock()`; a concurrent call returns 409.
11. **JSON safety.** Pass anything stored in JSON columns or returned to the client through `json_safe()` (NaN/inf to None, NumPy scalars to Python).
12. **Scoring weights** are in `engine/scorer.py` (`WEIGHTS`). The after-score applies a retention penalty for dropped informative columns so the score cannot be gamed by deleting columns.

## How to add a new cleaning fix (typical feature)

1. **Detect:** add an issue type in `engine/detector.py` (return `issue_type, column, severity, title, details`).
2. **Recommend:** map it to a `method_key` in `engine/rules.py` via `_rec(...)`; put every parameter the executor needs in `stats_context`.
3. **Explain:** add a deterministic fallback text for the `method_key` in `engine/explainer.py` (`generate_fallback_explanation`).
4. **Transform:** add a pure function in `engine/transforms.py`.
5. **Execute:** add a step in `engine/executor.py` at the correct position in the safe order, calling `apply("fn_name", "description", **kwargs)`.
6. **Score (if relevant):** update `engine/scorer.py` so the issue affects the right sub-score.
7. **Verify:** run the exported `pipeline.py` on the raw file and confirm it reproduces the exported CSV exactly.

Safe execution order: missing target -> duplicates -> identifiers -> datetime -> units -> multi-hot -> negatives -> column drops -> outlier capping -> imputation -> rare grouping -> one-hot.

## Conventions

**Backend**
- Heavy endpoints are synchronous `def` (run in the threadpool). Keep engine code free of FastAPI imports.
- Raise `HTTPException` in endpoints, domain errors (`BenchmarkError`, `DatasetParseError`) in engine/storage.
- Type hints on public functions; keep functions small and single-purpose.
- New DB columns must be nullable (auto-migration only does additive `ADD COLUMN`).
- Never log secrets or full request exceptions (only exception type names for Gemini calls).

**Frontend**
- All backend calls go through `src/services/api.ts`. Do not call `fetch` from components.
- Wizard state is owned by `src/app/page.tsx`; components stay presentational.
- Shared UI primitives are in `components/ui.tsx` (dashboard) and `components/primitives.tsx` + `lib/motion.ts` (landing).
- Respect `prefers-reduced-motion`.
- Keep TypeScript strict; run `npm run check-types` before finishing.
- If you change an API response shape, update both `backend/app/schemas/dataset.py` and the interfaces in `api.ts`.

## Configuration

Backend `.env` (see `backend/.env.example`): `DATABASE_URL`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_FALLBACK_MODELS`,
`MAX_UPLOAD_MB`, `MIN_ROWS`, `FILE_RETENTION_DAYS`, `BACKEND_CORS_ORIGINS`.
Frontend: `NEXT_PUBLIC_API_URL` (dashboard), `NEXT_PUBLIC_DASHBOARD_URL` (landing), `NEXT_PUBLIC_LANDING_URL` (dashboard header link).

## Security rules

- Never commit `.env`, `*.db`, or anything under `backend/uploads/` (except `.gitkeep`).
- Never send `.env` contents or user datasets to external services.
- Uploaded filenames must go through `sanitize_filename`; stored files are always `{uuid}.csv`.
- Client identity is `X-Client-Id` (isolation, not authentication). Do not treat it as auth.

## Known limitations (do not "fix" accidentally; discuss first)

- Diagnose/execute/benchmark are synchronous requests, not background jobs.
- Locks are in-process (not safe across multiple Uvicorn workers).
- Benchmark samples at most 2000 rows with default hyperparameters.
- No automated test suite in the repo yet; verification has been done with ad-hoc `TestClient` scripts.
- Local disk storage only.

## Working rules for agents

- Plan first for anything non-trivial: list the files you will touch and why, then wait for approval.
- Make small, focused changes. One logical change per commit, Conventional Commits style (`feat(engine): ...`, `fix(api): ...`).
- Follow existing patterns in the nearest sibling file before inventing a new one.
- Do not refactor unrelated code, rename public API fields, or add dependencies without asking.
- Add or update tests for engine logic when a test folder exists; otherwise describe how you verified the change.
- Never edit `ARCHITECTURE.md` or `README.md` unless the change affects documented behavior.