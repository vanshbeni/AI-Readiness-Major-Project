# Nebulus: Hardcoding & Improvement Audit

This document lists every place where Nebulus makes a fixed decision that a user, a dataset, or a config file should be able to influence, and what to do about each one. Line numbers refer to the code as of this audit.

## The short answer

Nebulus is **not** hardcoded to specific datasets. Nothing in the engine knows about Titanic, churn or housing; those names only exist in the demo generator (`backend/app/api/v1/endpoints/datasets.py:37-116`). A stress test with eight never-seen datasets (unicode column names, `dd/mm/yyyy` dates, `"12 ly"` unit strings, semicolon/Latin-1 CSV, Excel, 150 columns, 12 rows, text-only) ran end to end without a crash.

What **is** hardcoded is every *decision*:

| Layer | Adapts to the data? | Who controls the rules? |
|---|---|---|
| Column type inference | Yes, from values (names are hints) | Fixed regexes and ratios in code |
| Issue detection | Yes | Fixed thresholds in code |
| Health score | Yes | Fixed weights and penalty formulas |
| Fix selection | Partly (skew picks mean vs median) | One fixed method per issue, no alternatives |
| Pipeline order | No | Fixed 12-step order |
| Model benchmark | No | Fixed 5–7 models, fixed hyperparameters, fixed metric, 2,000-row cap |
| AI explanations | n/a | Fixed English prompt and fallback templates |
| Frontend | n/a | Duplicates several backend constants that can drift |

So the right mental model is: **the pipeline is generic, the policy is frozen.** The user can approve or reject a fix but cannot change *how* it is done, *when* something counts as a problem, or *what* "healthy" means.

## Correctness problems found during the audit

These are worse than "not configurable": they produce wrong output or false claims today.

| Status | Problem | Where |
|---|---|---|
| **Fixed** | One-hot and multi-hot indicator columns could **overwrite each other**: `safe_name()` maps `"Kepler"` / `"kepler "` and `"A+"` / `"A-"` to the same name, so the second category's indicator silently replaced the first. Names are now de-duplicated (`indicator_names`); the exported script inherits the fix. | `transforms.py:31-47, 112, 159` |
| **Fixed** | **Date features were never created.** pandas 3 reads text as `StringDtype`, but the detector checked `series.dtype == object` (always false), so date columns were never parsed and the benchmark later dropped them as high-cardinality text. A time-ordered regression went from R² −0.02 to 0.995 after the fix. | `detector.py:126-130` |
| **Fixed** | Mixed time-zone timestamps failed to parse and were one-hot encoded as junk; version strings (`1.2.3`) were typed as dates. Now `utc=True` everywhere, plus a version-string guard. | `profiler.py:18, 99-111`, `transforms.py:50-54` |
| **Fixed** | Generated columns (`X_year`/`_month`/`_day`, `X_value`/`_unit`, `has_X`) overwrote existing columns with the same name. Now suffixed via `fresh_name()`. | `transforms.py:21-28` |
| **Fixed** | Landing page claimed "destructive steps are off by default". Four of the six destructive fixes (`drop_identifier`, `drop_missing_target`, `drop_constant`, `drop_high_cardinality`) are approved by default; only "drop >70% missing" and "drop collinear" start off. Copy corrected. Defaults themselves are still a policy decision (see §4). | `rules.py:8`, `landing/.../Features.tsx:148` |
| **Fixed** | Landing page and leaderboard claimed imputation happens inside CV folds / "no peeking". Approved cleaning fixes (imputation, capping, rare grouping) are fitted on **all** rows before benchmarking; only scaling, SMOTE and leftover-NaN imputation are per fold. Copy corrected. Real fix: §5. | `Features.tsx:157`, `ModelLeaderboard.tsx` |
| **Fixed** | The "pipeline.log" terminal always printed "capping outliers" / "encoding categoricals" even when those fixes were switched off. It now lists only approved steps, in executor order. | `RecommendationChecklist.tsx` |
| Open | **Regression gets 15 free points**: target balance is always 100 for regression, so classification and regression scores aren't comparable. | `scorer.py:62-67` |
| Open | **Score cliff**: imbalance ratio 2.99 → balance 100; ratio 3.0 → 76 (detection starts at 3.0, penalty is `(ratio−1)×12`). | `detector.py:274`, `scorer.py:65` |
| Open | **Annual time series lose their year column**: an all-unique, consecutive-integer column with > 20 rows is typed `id` *before* the year check runs, so `Year = 1990…2023` is dropped as an identifier. | `profiler.py:63-66` vs `:78` |
| Open | **Type inference reads only the first 60 values**, so a file sorted by date or starting with 60 placeholder rows is misclassified. | `profiler.py:96` |
| Open | **Disguised missing values** (`"?"`, `"-"`, `"unknown"`, `-999`, `9999`) are treated as real data. | `storage.py:83` (default `na_values`) |
| Open | **Leading zeros lost**: zip / account codes like `00123` become `123` because pandas infers dtypes at ingest. | `storage.py:83, 122` |
| Open | **Day/month ambiguity**: `03/04/2021` is always read month-first (no `dayfirst`); on a `dd/mm/yyyy` column 77 of 200 rows got the wrong month. | `profiler.py:99-111`, `transforms.py:50-54` |
| Open | Years are mean-imputed (`year_range` is in `NUMERIC_TYPES`), producing values like `2011.37`. | `rules.py:3, 75` |
| Open | XGBoost and Gradient Boosting get **no class weighting** while the other models do, so the comparison is unequal when weights are enabled. | `benchmark.py:121, 130` |
| Open | Real cell values (`sample_values`, `majority_class`) are sent to Google with no privacy switch. | `rules.py:101, 108, 115` → `explainer.py:220-240` |

Severity tags used below:

- **[high]** silently produces wrong or misleading results on real data
- **[med]** limits usefulness or user control
- **[low]** cosmetic, or a maintainability risk

---

## 1. Health score (`backend/app/engine/scorer.py`)

| Where | Hardcoded | Problem | Improvement |
|---|---|---|---|
| `:3-10` | `WEIGHTS` = missingness 25%, the other five 15% each | A fraud dataset cares about balance far more than duplicates; a sensor dataset cares about outliers. One weighting for every domain. **[med]** | Move weights into a `ScoringProfile` (config + per-dataset override via API). Ship presets: `balanced`, `classification-heavy`, `regression-heavy`, `strict`. |
| `:13-22` | Grade bands 90/80/70/60 → A–F | Fine as a default, but not configurable and the label text is fixed English. **[low]** | Put bands + labels in the same profile. |
| `:41` | Missingness penalty `overall% × 2.5 + worst_col% × 0.25` | Magic multipliers with no justification shown to the user. A single 100%-empty optional column costs only 25 points. **[med]** | Document the formula in the UI; make multipliers part of the profile. |
| `:46` | Duplicates `dup% × 5` | 20% duplicates = score 0, regardless of whether duplicates are legitimate (e.g. repeated transactions). **[med]** | Let the user mark "duplicates are valid"; make the multiplier configurable. |
| `:52` | Outliers `avg_outlier% × 3 + n_columns × 4` | Penalises wide datasets: 25 columns with tiny outliers score 0. **[high]** | Normalise by number of numeric columns, not raw count. |
| `:59` | Validity `20 points per invalid column` | Validity only means "negative where name suggests non-negative" (see §2). Five such columns = 0. **[med]** | Add more validity checks (ranges, allowed values, regex patterns) and weight by affected-row %. |
| `:65` | Balance `(ratio − 1) × 12` | Ignores multi-class nuance (only majority/minority ratio). **[low]** | Use normalised entropy of the class distribution. |
| `:73` | Feature quality `10 / 15 / 5` per collinear / constant / ID issue | Same "count of issues" problem as outliers. **[med]** | Normalise by feature count. |
| `:87` | Retention penalty capped at 20 points | Reasonable guard, but the cap is arbitrary. **[low]** | Configurable. |
| `:94-107` | Driver thresholds (80/90/85/90/80/85) | Independent magic numbers that decide what appears in the summary. **[low]** | Derive from the profile. |
| `:112-117` | Summary sentences + 85/70 bands | English-only, fixed text. **[low]** | Template file / i18n. |

**Duplicated in the frontend:** `frontend/apps/dashboard/src/components/HealthScoreGauge.tsx:21-26` prints the weights ("·25%", "·15%") as literals. Change `WEIGHTS` in Python and the UI keeps lying. **[med]** → return weights in the API response and render them.

## 2. Issue detection thresholds (`backend/app/engine/detector.py`)

| Where | Hardcoded | Problem | Improvement |
|---|---|---|---|
| `:6-10` | `NON_NEGATIVE_KEYWORDS` (age, price, salary, fare, tenure, …) | **Negative-value validity only works if the column name is an English word from this list.** `Mass`, `Gewicht`, `poids`, `amt_usd` are never checked. **[high]** | Infer from data (column is ≥99% non-negative ⇒ negatives are suspect) and let users declare per-column rules (`min`, `max`, `allowed`, `regex`). |
| `:15` | `HIGH_CARDINALITY_LIMIT = 50` | A 60-value "country" column is dropped as free text; the user can't say "keep it, target-encode it". **[high]** | Make it relative to row count and offer alternative encodings (§5). |
| `:16` | `RARE_GROUPING_LIMIT = 10` | Keep-top-10 regardless of how much of the data the top 10 cover. **[med]** | Group by frequency share (e.g. categories < 1% → "Other"), configurable. |
| `:19` | High-cardinality ratio `> 0.40` | Magic. **[low]** | Profile setting. |
| `:88-95` | Missing severity bands 50 / 20 / 5 % | Fixed. **[low]** | Profile setting. |
| `:42` | Duplicate severity 5 / 1 % | Fixed. **[low]** | Profile setting. |
| `:203-211` | Outliers = **1.5 × IQR**, flagged only if > 1% of rows | Wrong for heavy-tailed data (income, prices, counts): flags thousands of legitimate values, then caps them. **[high]** | Offer IQR k (1.5/3.0), z-score, MAD, or "log-transform first"; skip capping for known heavy-tailed distributions (skew > 2). |
| `:250` | Multicollinearity `|r| > 0.85`, Pearson only | Misses non-linear redundancy; 0.85 is arbitrary. **[med]** | Configurable threshold; add Spearman / VIF. |
| `:254` | Only the top **5** correlated pairs are reported | Wide datasets with 30 redundant pairs only see 5. **[med]** | Report all above threshold (paginate in UI). |
| `:274-279` | Imbalance: ratio ≥ 3 flagged; ≥ 5 high; ≥ 10 or > 90% critical | Fixed. **[low]** | Profile setting. |
| `:115` | Constant feature = exactly 1 non-null unique value | A column that is 99.9% one value is never flagged; a 99%-NaN column with one value is called "constant" although *whether it is filled* may be signal. **[med]** | Dominant-value share threshold (e.g. ≥ 0.99); treat NaN as a value. |
| `:206` | Outliers skipped when IQR = 0 | Zero-inflated columns (85% zeros) never get checked. **[med]** | Fall back to MAD / percentiles. |
| `:239-242` | Pearson correlation also run on low-cardinality numeric *codes* | Correlating nominal codes is meaningless. **[low]** | Spearman for ordinal, Cramér's V for categoricals. |
| — | No checks for: inconsistent casing/whitespace (`"Kepler"` vs `"kepler "`), leading/trailing spaces, mixed types in one column, impossible dates, target leakage (feature ≈ target), near-constant columns (99% one value) | Real-world data problems that go undetected. **[high]** | Add these detectors (see roadmap). |

## 3. Column type inference (`backend/app/engine/profiler.py`)

| Where | Hardcoded | Problem | Improvement |
|---|---|---|---|
| `:7-8` | `ID_TOKENS`, `STRONG_ID_TOKENS` (English) | `Planet Désignation` or `numéro_client` not recognised as IDs (falls back to "all unique text", which is then dropped anyway). **[low]** | Rely on value patterns (unique + monotonic / fixed-width codes) first; names second. |
| `:9` | `BOOLEAN_STRINGS` (English yes/no/true/false) | `ja/nein`, `oui/non`, `sí/no` become categorical (still works, just labelled differently). **[low]** | Extendable list in config. |
| `:10-11` | `DATE_NAME_TOKENS`, `YEAR_NAME_TOKENS` | Year detection *requires* a year-like column name (`:78`). A column `Built` with 1950–2020 is treated as plain numeric. **[med]** | Detect year ranges from values alone (all ints in 1800–2100, low cardinality). |
| `:82` | Year range 1800–2100 | Historical/astronomical data excluded. **[low]** | Configurable. |
| `:87` | Numeric with ≤ 5 unique and < 5% unique ⇒ categorical | A 1–5 rating scale becomes categorical (one-hot) instead of ordinal. **[med]** | Let the user override any column's type in the UI (the single most useful control you can add). |
| `:99-111` | Date detection: ≥ 80% regex hits, or name hint + 50%, then 75% parse success (version strings excluded) | Reasonable, but invisible and unchangeable; month-first only. **[low]** | Per-column override. |
| `:114-117` | Unit pattern `^\$?\d+(\.\d+)?\s*[a-zA-Z%]+$` at ≥ 70% | **Misses `$12,345`, `1.234,56 €`, `12 345 km`, `€20`, negative values.** Such columns are typed as text and then dropped as high-cardinality. **[high]** | Add a currency/number parser (thousand separators, locale decimal comma, prefix/suffix symbols) and a `currency_numeric` type. The same pattern exists in `transforms.py:13`. |
| `:121` | Multi-label requires a delimiter in ≥ 40% of values **and** an English name token (`cast`, `genre`, `tag`, …) | A `skills` column works, `competences` doesn't. **[med]** | Drop the name requirement; use value structure (consistent delimiter, repeated tokens). |
| `:129-132` | Person-name detection by English name tokens and > 15 unique | Cosmetic; outcome is similar to "text". **[low]** | — |
| `:135` | Categorical if ≤ 25 unique or < 5% unique | Fixed. **[low]** | Profile setting. |

## 4. Remediation rule matrix (`backend/app/engine/rules.py`)

This is the biggest "hardcoded" area: **each issue maps to exactly one method**, and the user can only accept or reject it.

| Where | Hardcoded choice | What the user should be able to pick instead |
|---|---|---|
| `:70` | Missing > 70% ⇒ drop column (off by default) | Keep + "is_missing" flag; threshold slider |
| `:76-85` | Numeric missing: `|skew| > 1` ⇒ median, else mean | Median / mean / constant / KNN / iterative (MICE) / forward-fill for time series / add missing-indicator |
| `:86-90` | Categorical missing ⇒ mode | Mode / "Missing" category / constant |
| `:91-95` | Text missing ⇒ `"Unknown"` | — |
| `:119-124` | Invalid negatives ⇒ replace with median | Set to NaN then impute / absolute value / drop rows / clip at 0 |
| `:126-135` | Outliers ⇒ cap at 1.5 × IQR | Cap at custom percentiles / remove rows / log or Yeo-Johnson transform / leave alone |
| `:137-144` | Collinear pair ⇒ always drop the **second** column (`col2`) | Choose which to keep (e.g. the one more correlated with the target), or combine (PCA / ratio) |
| `:152-158` | High-cardinality text ⇒ drop | Target / frequency / hashing encoding; TF-IDF for real text |
| `:160-166` | Rare categories ⇒ top **10** + Other | Frequency threshold; keep-N slider |
| `:168-174` | Categorical ⇒ one-hot | One-hot / ordinal / target / frequency encoding |
| `:182-191` | Imbalance ratio ≥ **6** ⇒ SMOTE + weights, else weights | SMOTE / random over- or under-sampling / weights only / none |
| `:96-102` | Datetime ⇒ year, month, day only | Day-of-week, hour, is-weekend, elapsed days, cyclical encoding |
| — | No scaling choice (always StandardScaler inside training) | Standard / Robust / MinMax / none |

Also: `select_remediation_methods` accepts `problem_type` (`:24`) but never uses it, so regression and classification get identical rules. The approval API only accepts `Dict[str, bool]` (`schemas/dataset.py` `RecommendationApprovalRequest`), which is the structural reason no method or parameter can be changed.

**Improvement (architectural):** change each recommendation from `{method}` to `{method, alternatives[], params{}}` where `params` are editable (thresholds, k, percentiles). The executor already dispatches on `method_key`, so this is mostly a schema + UI change.

## 5. Pipeline execution (`backend/app/engine/executor.py`, `transforms.py`)

| Where | Hardcoded | Problem | Improvement |
|---|---|---|---|
| `executor.py:87-89` | Fixed 12-step order (target rows → dups → IDs → dates → units → multi-label → negatives → drops → capping → imputation → rare → encoding) | Sensible default, but e.g. imputing *before* capping, or deduplicating *after* type fixes, is impossible. **[low]** | Keep the order as the default; expose it as a declarative list. |
| `executor.py:12` | `MAX_ONE_HOT_CATEGORIES = 50` | Silent cap. **[med]** | Surface in UI / config. |
| `executor.py:13` | `MAX_MULTI_HOT_TOKENS = 15` | Only 15 genres/tags kept, silently. **[med]** | Configurable; show what was dropped. |
| `executor.py:139` | Unparseable dates filled with median year, fallback **2000** | Silent fabrication. **[med]** | Add an "is_missing_date" flag instead. |
| `executor.py:152, 173` | Fallback fill value **0.0** when a median can't be computed | Silent fabrication. **[med]** | Fail loudly or use NaN + flag. |
| `executor.py:203-204` | Outlier bounds fall back to 1st/99th percentile | Second hidden rule. **[low]** | Document / configurable. |
| `executor.py:237` | `keep_top` default 10 (duplicate of detector constant) | Two sources of truth. **[low]** | Single constant. |
| `transforms.py:13` | Same unit regex as profiler | Same currency gaps as §3. **[high]** | Shared parser. |
| `transforms.py:87-95` | Dates → year, month, day only | Hour, weekday, quarter, weekend flag, cyclical encoding and elapsed time are lost (for taxi pickups, hour is the main signal). **[med]** | Configurable date parts + `dayfirst`. |
| `transforms.py:14` | Tag split on `,` `;` `\|` only | `"Action/Adventure"` isn't split; `"Smith, John"` is. Tokens are case-sensitive. **[low]** | Configurable delimiter, case-folding. |
| `executor.py:71` | Script reads input with plain `pd.read_csv` | The user's original semicolon / Windows-1252 / Excel file won't load in the exported script, because ingest normalisation isn't reproduced. **[med]** | Embed the detected delimiter / encoding in the script. |
| — | Cleaning statistics (medians, bounds, rare lists) are fitted on all rows, then the benchmark cross-validates on the cleaned file | Mild leakage between folds for imputation and capping. **[med]** | Export a fitted scikit-learn `Pipeline`/`ColumnTransformer` and fit it inside each fold. |

## 6. Model benchmarking (`backend/app/engine/benchmark.py`)

| Where | Hardcoded | Problem | Improvement |
|---|---|---|---|
| `:19, :97` | `MAX_BENCHMARK_ROWS = 2000` | **Benchmarks train on at most 2,000 rows, whatever the dataset size.** A 500k-row dataset is judged on 0.4% of its data, and the leaderboard doesn't say so. **[high]** | Make configurable; show "trained on N of M rows" in the UI; offer a "full run" in a background job. |
| `:20` | `MIN_BENCHMARK_ROWS = 10` | OK. **[low]** | — |
| `:21, :53` | Text columns with > 50 categories dropped from training | Silent. **[med]** | Report dropped columns in the leaderboard response. |
| `:107, :141` | Folds `min(5, …)`, fixed | No user choice; no time-series split for temporal data (causes leakage). **[high for time series]** | Configurable k; `TimeSeriesSplit` / `GroupKFold` options. |
| `:109` | Classification metric = **macro F1** only | No ROC-AUC, PR-AUC, accuracy, balanced accuracy, log-loss. Fraud/medical users need recall or PR-AUC. **[high]** | Metric picker; compute several and rank by the chosen one. |
| `:143` | Regression metric = **R²** only | No MAE / RMSE / MAPE; R² is hard to explain to business users. **[med]** | Same as above. |
| `:117-166` | Fixed model list and fixed hyperparameters (`n_estimators=100`, `max_depth=8/5/3`, `alpha=1.0`, `kernel="rbf"`, …) | No tuning; deep models capped shallow; SVC is slow on large data. **[med]** | Model picker + optional small randomized search; per-model time budget. |
| `:68, :108, :142, …` | `random_state=42` everywhere | Reproducible (good) but users can't test stability. **[low]** | Seed setting; optional repeated CV with mean ± std. |
| `:101-114` | SMOTE `k_neighbors = 5` (shrunk for tiny classes) | Fine. **[low]** | — |
| `:187` | Display `x% ` for classification, 4 decimals for regression | — **[low]** | — |
| `:180` | Only the mean CV score is kept | The standard deviation is discarded, so a 0.81 vs 0.80 "win" may be noise. **[med]** | Report mean ± std per model. |
| — | No dummy baseline | "F1 = 0.52" can't be judged against a trivial predictor. **[med]** | Add `DummyClassifier`/`DummyRegressor` as rank 0. |
| `:128-138` | XGBoost / LightGBM only if installed | The leaderboard differs between environments. **[low]** | Show which optional models were skipped. |
| `:117, 130, …` | `n_jobs=-1` everywhere | Each request grabs every core; concurrent users fight. **[med for deployment]** | Configurable worker count. |
| `schemas/dataset.py:193` | `cv_folds: int = 3` default | Disagrees with the engine's 5. **[low]** | Single source of truth. |
| — | No feature importance, confusion matrix, residual plots, calibration | The leaderboard says *who* won, not *why*. **[med]** | Return importances + per-fold scores; visualise in the visual lab. |

## 7. AI explanations (`backend/app/engine/explainer.py`, `core/config.py`)

| Where | Hardcoded | Problem | Improvement |
|---|---|---|---|
| `config.py:26` | Fallback list `["gemini-3.7-flash", "gemini-3.8-flash"]` repeats the primary model | Only one real fallback. **[low]** | De-duplicate at load time. |
| `config.py:24-30` | Model names, timeout 30 s, 2 retries, 45 s total budget | Diagnostics can block for ~45 s when Gemini is overloaded (observed during testing). **[med]** | Run explanations **asynchronously** after diagnostics return; stream them in. |
| `explainer.py:231` | English system prompt "You are a Senior Data Scientist…" | Fixed language, tone and length. **[low]** | Prompt template file; language + audience (student / analyst / exec) setting. |
| `explainer.py:241` | `temperature: 0.2` | Fine. **[low]** | Config. |
| `explainer.py:139` | Retryable statuses `{429, 500, 502, 503, 504}` | Fine. **[low]** | — |
| `explainer.py:20-110` | ~15 English statistical fallback templates | Text is duplicated knowledge of the thresholds (e.g. "above the 70% viability threshold" at `:40`, "1.5x IQR" at `:89`). Change a threshold and the text lies. **[med]** | Fill thresholds into the text from the same config. |
| — | Only Gemini | Vendor lock-in; free-tier quota exhaustion seen in testing. **[med]** | Provider interface (Gemini / OpenAI / local Ollama). |

## 8. Objective & target suggestions (`backend/app/api/v1/endpoints/objectives.py`)

| Where | Hardcoded | Problem | Improvement |
|---|---|---|---|
| `:53` | Numeric target with ≤ 2 unique, or integer with ≤ 10 unique ⇒ classification | A 1–20 integer rating target is suggested as regression; a 0–10 count target as classification. Suggestion only, user can override. **[low]** | Show the reason (already returned) more prominently. |
| `:57` | Text target with > max(20, 50%) unique ⇒ unsuitable | Fine. **[low]** | — |
| `:77` | Only `classification` / `regression` accepted | See §11. **[med]** | — |
| `:25, :38` | ≥ 10 target values, ≥ 2 rows per class | Correct guard rails. | — |
| `frontend/.../ObjectiveStep.tsx:13` | `TARGET_NAME_HINTS` includes `survived`, `churn`, `medianhousevalue` | **The only place where demo-dataset names leak into product logic.** Harmless (only pre-selects a default) but looks hardcoded. **[low]** | Remove demo names; rely on the backend's `target-suggestions` scores. |

## 9. Config, storage & API (`backend/app/core/*`, `api/*`)

| Where | Hardcoded | Problem | Improvement |
|---|---|---|---|
| `config.py:44-47` | CORS origins = `localhost:3000` only | Breaks any deployment unless overridden in `.env`. **[med]** | Document `BACKEND_CORS_ORIGINS` in `.env.example`; read from env in production. |
| `config.py:39` + `frontend/.../api.ts:4` | `MAX_UPLOAD_MB = 200` in **both** backend and frontend | Two sources of truth. **[low]** | `GET /config` endpoint the frontend reads at startup. |
| `storage.py:20` + `api.ts:5` | Supported extensions duplicated | Same. Also no Parquet / JSON / TSV-by-extension. **[low]** | Same endpoint; add Parquet + JSON readers. |
| `config.py:40` | `MIN_ROWS = 10` | Fine. | — |
| `config.py:41` | 7-day file retention | Fine, configurable. | — |
| `storage.py:21` | CSV encodings tried: utf-8-sig, utf-8, cp1252, latin-1 | cp1252/latin-1 decode almost anything, so a wrong guess can produce mojibake silently. **[low]** | Use `charset-normalizer` and warn when confidence is low. |
| `storage.py:90` | Excel: first non-empty sheet only | Multi-sheet workbooks lose data (a warning is shown). **[low]** | Sheet picker. |
| `storage.py:181-187` | In-process dataset locks | Breaks with multiple workers / servers. **[med for deployment]** | Redis or DB-row locks. |
| `deps.py:19-22` | Ownership via random `X-Client-Id` in `localStorage` | Not authentication: clearing the browser loses all datasets; sharing a link doesn't work. **[med]** | Real auth (even magic-link email) or signed share tokens. |
| `datasets.py:112-116` | Three synthetic demo datasets generated with fixed seeds | Fine for demos; real public datasets would be more convincing. **[low]** | Optional download of real Titanic/Adult/etc. |
| — | Everything runs synchronously in the request | Large files block a worker for minutes. **[med]** | Background jobs (RQ/Celery/arq) + progress via SSE. |

## 10. PDF report (`backend/app/engine/reporter.py`)

Layout, colours (`#0284c7`, `#cbd5e1`, …), column widths (`:135`), section order and every sentence are fixed in code, and the palette no longer matches the redesigned UI. More specifically:

- Helvetica has no CJK / Devanagari glyphs, so Hindi or Chinese column names render as empty boxes.
- US Letter only; title and footer say "AI Data Readiness Platform", not Nebulus.
- The `summary_stats` parameter (`:16`) is accepted but never used: row/column counts, missing cells and type breakdown are missing from the PDF, as are the issue list, recommendations and column profiles.
- Only the top 5 models are listed.

**[low–med]** → Move to an HTML template (Jinja2) rendered with WeasyPrint, so the report can reuse the dashboard's design and include charts from the visual lab.

## 11. Frontend (`frontend/apps/dashboard/src`)

| Where | Hardcoded | Problem | Improvement |
|---|---|---|---|
| `services/api.ts:1`, `Header.tsx:12` | `localhost:8000` / `localhost:3001` fallbacks | Fine for dev; must be set via env in production. **[low]** | — |
| `HealthScoreGauge.tsx:21-26` | Weight labels | Duplicates `scorer.WEIGHTS` (§1). **[med]** | Send weights from the API. |
| `RecommendationChecklist.tsx:17-27`, `Terminal.tsx:55-66` | Terminal progress lines and model names are **scripted animations**, not real progress | If the backend changes order or models, the UI shows fiction. **[med]** | Real progress events via SSE (`/datasets/{id}/execute/stream`). |
| `components/Stepper.tsx` | Fixed 5-step linear flow | Can't skip to the visual lab before choosing a target, or benchmark without cleaning. **[low]** | Visual lab is now reachable from any step via the header. |
| all components | English-only copy | **[low]** | i18n (next-intl). |
| `ObjectiveStep.tsx:13` | Demo names in `TARGET_NAME_HINTS` | See §8. **[low]** | — |
| several | Five different banding schemes for "good/bad": grades 90/80/70/60 (`scorer.py:13`), summary 85/70 (`scorer.py:112`), gauge colours 85/65/50 (`HealthScoreGauge.tsx`), missing colour 20 (`ProfileTable.tsx`), landing scanner 85/70 | Inconsistent signals to the user. **[low]** | One shared band definition from `GET /config`. |
| `UploadStep.tsx` `DEMOS` | Demo keys, file names and descriptions repeated from `datasets.py:112-116` ("~300 rows"; actual 304/305) | Drift (the churn ratio text has already drifted once). **[low]** | Serve demo metadata from the API. |
| step labels | "01/05" … "05/05" hard-written in five components | Adding a step means editing six places. **[low]** | Derive from `STEPS`. |
| `BeforeAfterDiff.tsx` | "before" always red, "after" always green | Wrong colour when a metric got worse. **[low]** | Colour by direction. |
| `services/api.ts` types | Hand-copied from Pydantic schemas | Already drifted (`numeric_stats` lacks `q25`/`q75`). **[low]** | Generate from OpenAPI. |
| — | No way to override a column's inferred type | The most requested control in tools like this. **[high]** | Type dropdown per column in the profile table → re-run diagnostics. |

## 12. Visual lab constants (new, `backend/app/engine/visualizer.py`)

The new visual lab has its own (documented) defaults:

- 2,000-row random sample for charts (`max_rows`, adjustable per request up to 10,000); stats, correlations and PCA use the full data.
- Up to 60 columns sent (`max_cols`), with a quota so categorical (15) and date (5) columns aren't crowded out by numeric ones.
- Top 15 categories per column; text previews truncated to 60 characters.
- PCA fitted on up to 50,000 rows; k-means tries k = 2–6 and keeps the best silhouette.
- Correlation limited to the 40 least-missing numeric columns.

**Improvement:** expose sample size and k as controls in the lab's UI.

## 13. Capabilities that are effectively "hardcoded away"

- **Problem types:** only binary/multi-class classification and regression. No clustering, time-series forecasting, multi-label, multi-output or anomaly detection.
- **Time series:** no datetime index awareness, no lag features, random CV leaks the future.
- **Text:** free text is dropped. No TF-IDF / embeddings.
- **Leakage detection:** no check for features that are near-copies of the target or post-outcome columns.
- **Custom rules:** no way to say "`age` must be 0–120" or "`email` must match a regex".
- **Re-use:** the cleaning recipe can't be saved and applied to next month's file from the UI.
- **Comparison:** no side-by-side of two datasets or two pipeline runs.
- **Visualisation:** *(addressed by the new visual lab)*.

---

## 14. Recommended architecture change: one policy object

Most items above share a single fix: replace scattered module constants with one versioned policy that every stage reads.

```python
# backend/app/core/policy.py
class DetectionPolicy(BaseModel):
    missing_severity: list[float] = [5, 20, 50]
    outlier_method: Literal["iqr", "zscore", "mad"] = "iqr"
    outlier_k: float = 1.5
    collinearity_threshold: float = 0.85
    high_cardinality_limit: int = 50
    rare_category_share: float = 0.01
    # ...

class ScoringPolicy(BaseModel):
    weights: dict[str, float] = {"missingness": .25, "duplicate": .15, ...}
    grade_bands: list[tuple[float, str]] = [(90, "A"), (80, "B"), ...]

class BenchmarkPolicy(BaseModel):
    max_rows: int = 2000
    cv_folds: int = 5
    metric: str = "f1_macro"
    models: list[str] = ["rf", "logreg", "gb", "dt", "svc", "xgb", "lgbm"]

class Policy(BaseModel):
    preset: Literal["balanced", "strict", "lenient"] = "balanced"
    detection: DetectionPolicy = DetectionPolicy()
    scoring: ScoringPolicy = ScoringPolicy()
    benchmark: BenchmarkPolicy = BenchmarkPolicy()
```

- Store a `Policy` per dataset (JSON column), defaulting to the preset.
- `GET/PUT /datasets/{id}/policy` lets the UI edit it; changing it invalidates downstream results (the same mechanism objectives already use).
- `GET /config` returns global limits (upload size, extensions, weights) so the frontend stops duplicating them.
- Explanation templates read thresholds from the policy so the text always matches the behaviour.

## 15. Prioritised roadmap

| # | Improvement | Impact | Effort |
|---|---|---|---|
| 0 | Close the open correctness items at the top of this document (regression score bias, ratio-3 cliff, year-as-ID, first-60-values sampling, disguised NAs, leading zeros, `dayfirst`) | High | Low–Med |
| 1 | Show "benchmarked on N of M rows" and make `MAX_BENCHMARK_ROWS` configurable | High | Low |
| 2 | Currency / thousand-separator / locale number parsing | High | Low |
| 3 | Per-column type override in the profile table | High | Medium |
| 4 | Data-driven non-negative check instead of English keyword list | High | Low |
| 5 | Inconsistent-casing / whitespace category detector + "normalise" fix | High | Low |
| 6 | Metric picker (ROC-AUC, PR-AUC, recall, MAE, RMSE) | High | Medium |
| 7 | Alternative methods + editable params per recommendation | High | Medium |
| 8 | Async AI explanations (don't block diagnostics on Gemini) | Medium | Medium |
| 9 | `Policy` object + presets (§14) | High | High |
| 10 | `GET /config` to remove frontend/backend duplicated constants | Medium | Low |
| 11 | Normalise outlier and feature-quality penalties by column count | Medium | Low |
| 12 | Target-leakage detector | High | Medium |
| 13 | Real progress events (SSE) instead of scripted terminals | Medium | Medium |
| 14 | Time-series mode (TimeSeriesSplit, lag features) | Medium | High |
| 15 | Save & re-apply a cleaning recipe to a new file | Medium | Medium |
