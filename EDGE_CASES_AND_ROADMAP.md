# Nebulus: Edge Cases, Open Issues, Hardcoding & the Road to a One-Stop EDA Platform

**Scope.** This document answers four questions:

1. Which edge cases does Nebulus still get wrong?
2. Which issues can we fix, and how?
3. What is hardcoded today, *why* was it hardcoded, and how do we make it configurable?
4. What features turn Nebulus from a "clean-then-benchmark" wizard into a one-stop data analysis and EDA platform?

**How it was produced.** Every edge case marked *observed* was reproduced by running a real dataset through the same code path the API uses (ingest → objective → diagnose → execute → benchmark → visual lab). The 41-case battery is checked in at `backend/tests/edge_case_battery.py`, so every claim here can be re-run:

```powershell
cd backend
venv\Scripts\python tests\edge_case_battery.py                    # all cases
venv\Scripts\python tests\edge_case_battery.py dayfirst_dates     # one case
```

Hardcoded values were verified by reading the code; line-level detail lives in `HARDCODED_AUDIT.md`. This document explains the *why* and the *fix*.

**Legend.**

- **Status:** ✅ handled · 🔧 fixed while writing this document · ❌ open
- **Impact:** **H** = silently wrong results · **M** = limits usefulness · **L** = cosmetic or maintainability
- **Effort:** S = hours · M = days · L = weeks

---

## Part 1 — Executive summary

| Area | Today | Verdict |
|---|---|---|
| Robustness | 41 hostile datasets, **0 crashes** | Strong. Errors are caught and explained. |
| Correctness | 5 silent data bugs found; **4 fixed in this pass** (below), many semantic gaps remain | Medium. "Doesn't crash" ≠ "right answer". |
| Configurability | Every threshold, weight, fix method, model and metric is a constant | Weak. The pipeline is generic; the *policy* is frozen. |
| EDA coverage | Profile table, health score, issue list, and a 9-view visual lab | Partial. Covers ~30% of a full EDA workflow. |
| Modelling | Classification + regression, 5–7 fixed models, one metric each, 2,000-row cap | Basic. |
| Platform | Single user per browser, synchronous jobs, local disk, English only | Prototype. |

### Bugs fixed while writing this document

| Bug | Evidence | Fix |
|---|---|---|
| **Date features were never created.** pandas 3 reads text as `StringDtype`; the detector checked `series.dtype == object`, which is now always false. Every date column was skipped by cleaning and then dropped by the benchmark as "high-cardinality text". | Time-ordered regression: best R² **−0.02 → 0.995** after the fix. | `detector.py`: check "not numeric and not datetime" instead of `object`. |
| **Mixed time zones broke date parsing.** `"…+05:30"` next to `"…Z"` made `to_datetime` fail, so the column fell back to text and was one-hot encoded into 27 junk columns (`ts_2024_03_01t10_00_00z`, …). | `mixed_timezones` case. | `utc=True` in profiler, transforms and visualizer. |
| **Version strings were treated as dates.** `"1.2.3"` parsed as 2003-01-02, so once date extraction worked it would have produced fake year/month/day features. | `version_strings` case. | Version-pattern guard in type inference. |
| **Generated columns overwrote existing ones.** `has_Cabin`, `Date_year`, `X_value`/`X_unit` silently replaced user columns with the same name. | Direct transform test: `[5, 7, 9]` → `[1, 0, 1]`. | `fresh_name()` adds `_2`, `_3` suffixes. |
| *(previous pass)* One-hot / multi-hot columns overwrote each other (`"Kepler"` vs `"kepler "`). | `collision_test`. | `indicator_names()`. |

All fixes live in `transforms.py`, which is embedded in the exported pipeline script, so downloaded scripts get them too. The script still reproduces the cleaned CSV exactly.

---

## Part 2 — Edge cases: what is handled, what is not

### 2.1 Already handled ✅

These were tested and behave correctly. They're listed so nobody "fixes" them twice.

| Scenario | Behaviour |
|---|---|
| Empty file, header only, fewer than 10 rows, fewer than 2 usable columns | Rejected at upload with a clear message |
| All-empty rows or columns | Dropped at ingest |
| Duplicate or blank column names | Renamed (`name_2`, `column_3`) with a warning |
| `inf` / `-inf` | Converted to missing at ingest |
| Semicolon, tab and pipe delimiters; Latin-1 / cp1252 encodings | Sniffed and decoded |
| Excel `.xlsx` / `.xls` | Parsed (first non-empty sheet) |
| Target with a class of one row | Rejected at objective ("every class needs 2 or more rows") |
| Target with missing labels | Issue raised; "drop rows without a target" fix |
| Text target with many classes (30) | Accepted |
| Too few labelled rows to benchmark; one class left after cleaning | Clear 400 error |
| Duplicates that differ only by an ID column | Detected (IDs excluded from the comparison) |
| Unicode / emoji column names and categories | Work (after the collision fix) |
| Whitespace-padded numbers `"  12 "` | Parsed as numeric |
| Concurrent operations on the same dataset | 409 "another operation is running" |
| Page refresh mid-flow | Session restored from `?dataset=` |
| Gemini overloaded / quota exhausted / no key | Retries, fallback models, then statistical explanations |
| 800 columns × 150 rows; 150,000 rows × 5 columns | Complete (see timings in §2.6) |

### 2.2 Ingest & parsing ❌

| # | Scenario (observed) | What happens today | Impact | Proposed solution | Effort |
|---|---|---|---|---|---|
| I1 | Numbers with thousands separators: `"1,234"` | Typed **text** → dropped as high-cardinality | H | Locale-aware number normaliser at ingest: strip `,` / `_` / thin spaces when every value parses afterwards | S |
| I2 | European decimals with `;` delimiter: `"12,50"` | Typed **text** → dropped | H | Detect decimal comma when the delimiter isn't `,` and ≥ 90% of values match `^\d+,\d+$`; pass `decimal=","` | S |
| I3 | Currency: `"₹12,345"`, `"$1,200"`, `"€50"`, `"USD 100"` | Typed **text** → dropped | H | Currency parser: symbol/ISO code prefix or suffix → `amount` column (+ `currency` column if mixed) | S |
| I4 | Disguised missing values: `"?"`, `"-"`, `"N.A."`, `"unknown"` | Column becomes **text**, missing count = 0 | H | Configurable NA-token list; a "suspected placeholder" issue when one token dominates an otherwise numeric column | S |
| I5 | Sentinel numbers: `-999`, `9999`, `0` for "unknown" | Treated as real values; flagged as invalid negatives / outliers instead | H | Sentinel detector: a repeated round value far outside the distribution → offer "convert to missing" | S |
| I6 | Leading zeros: zip / account `"00123"` | Becomes `123` (int) after storage | H | Read with `dtype=str` first, then infer; never auto-convert strings with leading zeros | S |
| I7 | Title / metadata rows above the header | **Rejected**: "Error tokenizing data" | M | Header-row detection (first row whose field count matches the modal count); skip-rows option in the UI | S |
| I8 | UTF-16 CSV (Excel "Unicode text") | **Rejected**: "0 usable columns" | M | `charset-normalizer` for encoding detection; try UTF-16 when a BOM is present | S |
| I9 | Excel workbook whose first sheet is a README | **Rejected** (uses the README sheet) | M | Sheet picker; default to the largest sheet, not the first | S |
| I10 | Excel serial dates (`44927`) | Typed **numeric** | M | Numeric column named like a date with values in 20,000–60,000 → suggest "convert from Excel serial" | S |
| I11 | Multi-row / merged Excel headers | Header becomes the first row only | M | Header depth option; join multi-row headers with `_` | M |
| I12 | Very large files (close to 200 MB) | Whole file read into memory, decoded, re-written, re-parsed | M | Stream to disk; parse with DuckDB or Polars; sample for profiling | M |
| I13 | Parquet, JSON, TSV extension, SQL databases, Google Sheets | Not supported | M | Connector layer (see Part 5) | M–L |
| I14 | Multiple related files (orders + customers) | One file per dataset; no joins | M | Multi-table workspace with join builder | L |

### 2.3 Type inference ❌

| # | Scenario | What happens today | Impact | Proposed solution | Effort |
|---|---|---|---|---|---|
| T1 | Annual series `Year = 1990…2023`, one row per year (observed) | Typed **id** → dropped | H | Run the year check before the sequential-integer ID rule; never treat year-named columns as IDs | S |
| T2 | All-unique continuous floats named `…_index` (observed: `consumer_price_index`) | Typed **id** → dropped (name ends with "index") | H | ID rule must require integer or string codes, never floats | S |
| T3 | Placeholder values in the first 60 rows (observed: 70 × `"pending"` then dates) | Typed **text** (inference reads only the first 60 values) | H | Random seeded sample of 500 non-null values instead of `head(60)` | S |
| T4 | Day-first dates `dd/mm/yyyy` (observed) | Parsed month-first: **77 / 200 rows get the wrong month** | H | Detect day-first (any first field > 12 ⇒ day-first for the column); per-dataset locale setting | S |
| T5 | Mixed types: `"12"`, `"twelve"`, `"12kg"` | **text** → dropped | M | "Mixed-type column" issue showing the share of numeric values; offer "coerce to number, rest → missing" | S |
| T6 | Negative numbers with units: `"-5 °C"` | **text** | M | Extend the unit regex: sign, `°`, unit-before-number | S |
| T7 | Comma lists under non-English names (observed: `ingredients`) | **text** → dropped | M | Detect multi-label from structure (consistent delimiter + token reuse), drop the English name gate | S |
| T8 | Slash-delimited tags `"Action/Drama"` | **categorical** → rare grouping | M | Add `/` to delimiters when values look like tags | S |
| T9 | Ordinal categories `Low < Medium < High` | One-hot (order lost) | M | Ordinal detector for known scales + user-defined order | M |
| T10 | Integer codes with 6–25 values (`region_code`) | Treated as continuous: mean-imputed, IQR-capped | M | Per-column type override in the UI (fixes most of this table at once) | M |
| T11 | Non-English booleans `si/no`, `ja/nein`, `haan/nahi` | Categorical (works, mislabelled) | L | Extendable boolean pairs | S |
| T12 | Semantic types (email, phone, URL, IP, lat/long, postal code) | Generic text / numeric | M | Semantic type layer (regex + checksums); powers PII detection and geo maps | M |

### 2.4 Data-quality detection ❌

| # | Scenario | What happens today | Impact | Proposed solution | Effort |
|---|---|---|---|---|---|
| Q1 | Inconsistent spelling `"Delhi"`, `"delhi "`, `"DELHI"` (observed) | 5 separate categories, no issue | H | Normalisation issue: case-fold, trim, accent-fold, plus fuzzy near-duplicates (Levenshtein ≤ 1) → "merge variants" fix | S |
| Q2 | Same problem in the **target** `"Yes"`, `"yes "` (observed) | Treated as **4 classes** | H | Same normaliser, run on the target before validation, with a warning | S |
| Q3 | Target leakage: a feature that copies the target (observed) | No issue; leaderboard shows **100%** | H | Leakage detector: single-feature AUC/R² > 0.98, or a deterministic mapping to the target → critical issue, default "drop" | S |
| Q4 | Near-constant column (99.5% one value) (observed) | Not flagged | M | Dominant-value share ≥ 0.99 → "near-constant" issue | S |
| Q5 | Zero-inflated column with extreme spikes (observed) | Outliers **not** checked (IQR = 0) | M | Fall back to MAD / non-zero IQR when IQR = 0 | S |
| Q6 | Negatives in a non-negative column with an unknown name (observed: `mass_kg`) | Not flagged | M | Data-driven rule: ≥ 99% of values ≥ 0 ⇒ negatives are suspect; editable per-column min/max | S |
| Q7 | Legitimately negative column whose name matches the keyword list (observed: `net_income`) | "Fixed" to the median (destroys data) | H | Same data-driven rule; keywords become hints only, never sole triggers | S |
| Q8 | PII (emails, phones, national IDs) (observed) | Dropped as high-cardinality text, never flagged | H | PII detector + masking / hashing fix; never send PII to the LLM | S |
| Q9 | Predictive high-cardinality feature (observed: 80 zip codes that perfectly predict the target) | **Dropped by default**; best model 53% | H | Offer target / frequency encoding instead of dropping; never auto-approve dropping a feature correlated with the target | M |
| Q10 | Heavy-tailed numeric (income, population) | 5–10% of rows capped as outliers | H | Skew-aware detection (log-scale IQR when skew > 1) and a "transform instead of cap" option | S |
| Q11 | Impossible combinations (`end_date < start_date`, `age < years_employed`) | Not checked | M | Cross-column rule engine (Part 4) | M |
| Q12 | Near-duplicate rows (typos, whitespace) | Only exact duplicates found | M | Normalise, then hash; optional fuzzy dedupe for small data | M |
| Q13 | Drift between two files (train vs new month) | Not supported | M | Dataset comparison (PSI / KS per column) | M |
| Q14 | Class imbalance across many classes (one rare class among 20) | Whole target marked "critical" | L | Effective-number-of-classes / entropy-based severity | S |

### 2.5 Cleaning & transformation ❌

| # | Scenario | What happens today | Impact | Proposed solution | Effort |
|---|---|---|---|---|---|
| C1 | Year column with missing values (observed) | Mean-imputed → `2011.37` | M | Years: median, rounded | S |
| C2 | Every issue has exactly one fix method | Approve or reject only | H | Method alternatives + editable parameters (Part 4) | M |
| C3 | Four of six destructive fixes approved by default (drop IDs, constant, high-cardinality, missing-target rows) | Data removed unless the user notices | M | Policy setting: "destructive fixes start off"; never auto-drop target-correlated features | S |
| C4 | Fixed 12-step order | Can't impute before capping, or dedupe after normalising | L | Declarative step list with dependency checks | M |
| C5 | Date parts are year / month / day only | Hour, weekday, quarter, weekend, cyclical and elapsed time lost | M | Configurable date-part extractor | S |
| C6 | Silent caps: 50 one-hot categories, 15 multi-hot tokens | Extra categories quietly ignored | M | Report what was dropped; configurable caps | S |
| C7 | Collinear pair → always drops the second column in file order | Might drop the more predictive one | M | Drop the one with lower target correlation / higher missingness; let the user pick | S |
| C8 | Cleaning statistics fitted on all rows, then cross-validated | Mild fold leakage in benchmark scores | M | Export a fitted scikit-learn `Pipeline` and fit it inside each fold | L |
| C9 | No undo, no history, no manual edits | Mistakes require re-running everything | M | Versioned dataset history with diff + revert | M |
| C10 | The cleaning recipe can't be applied to next month's file | Only via the downloaded script | M | "Apply saved recipe" to a new upload, with schema-drift checks | M |
| C11 | Exported script reads input with plain `pd.read_csv` | Fails on semicolon / cp1252 / Excel originals | M | Embed the detected delimiter / encoding / sheet in the script | S |

### 2.6 Targets, modelling & scale ❌

| # | Scenario | What happens today | Impact | Proposed solution | Effort |
|---|---|---|---|---|---|
| M1 | 1% fraud (observed) | Ranked by macro F1 only; models trained on **2,000 of 3,000** rows | H | Metric picker (PR-AUC, recall@k, ROC-AUC); configurable row cap; show "trained on N of M" | M |
| M2 | 150k-row dataset (observed) | Benchmark uses **2,000 of 150,000** rows (1.3%) | H | Adaptive sample (time budget), full run as a background job | M |
| M3 | Time-ordered data | Random K-fold leaks the future | H | Time-series mode: `TimeSeriesSplit`, lag features, no shuffling | M |
| M4 | Skewed regression target (observed) | No log-transform suggestion; R² only | M | Target-transform suggestion (log / Box-Cox) + MAE / RMSE / MAPE | S |
| M5 | No baseline model | "F1 = 0.49" can't be judged | M | Dummy baseline at rank 0 | S |
| M6 | Mean CV score only | Noise vs real difference unknown | M | Mean ± std, per-fold scores | S |
| M7 | XGBoost / Gradient Boosting get no class weights | Unequal comparison | M | Pass `sample_weight` / `scale_pos_weight` | S |
| M8 | Only classification and regression | No clustering, forecasting, anomaly detection, multi-label | M | Task registry (Part 5) | L |
| M9 | No explainability | Users see *who* won, not *why* | M | Permutation importance, SHAP, partial dependence | M |
| M10 | Regression gets 15 free health points; balance score cliff at ratio 3.0 | Scores not comparable | M | Re-weight over applicable dimensions; continuous balance score | S |
| P1 | 800 columns: diagnose 19.7 s, benchmark 20.2 s (observed) | Synchronous HTTP request | M | Background jobs + progress events; cache inferred types (inference currently runs several times per column) | M |
| P2 | Diagnostics wait up to 45 s on Gemini | Whole request blocked | M | Explanations generated asynchronously and streamed in | M |
| P3 | Multiple server workers | In-process locks don't share | M | Redis or DB-row locks | S |

### 2.7 Security, privacy & multi-user ❌

| # | Scenario | Today | Impact | Solution | Effort |
|---|---|---|---|---|---|
| S1 | Clearing browser storage | All datasets lost (ownership = random `localStorage` ID) | M | Real accounts (magic link / OAuth) | M |
| S2 | Sharing a result with a teammate or professor | Impossible | M | Signed read-only share links | S |
| S3 | Sending data to Gemini | Real cell values (`sample_values`, `majority_class`) leave the machine | H | Privacy switch (default off for samples), PII redaction, local-LLM option | S |
| S4 | Deployment | CORS fixed to localhost; local disk; SQLite default | M | Env-driven config, object storage, Postgres (already supported) | S |
| S5 | Expired files | Cleanup only at startup; DB rows left behind (410 errors) | L | Scheduled cleanup that also removes rows | S |

---

## Part 3 — Everything that is hardcoded, and *why*

"Why" matters: most constants were chosen deliberately for an MVP, and the right fix differs by reason. A statistical convention should become a **default with an override**; a performance guard should become **adaptive**; a demo shortcut should be **deleted**.

**Reason codes:**

- **CONV** — statistical rule of thumb (defensible default, wrong for some domains)
- **LIB** — library default copied as-is
- **PERF** — guard against slow requests / large payloads
- **MVP** — simplest thing that worked; no UI existed to change it
- **SAFE** — prevents invalid input / crashes
- **REPRO** — reproducibility
- **DEMO** — tuned for the bundled demo datasets
- **UI** — presentation choice
- **DUP** — a copy of a value that lives elsewhere (drift risk)

**Target fix codes:**

- **POL** — per-dataset policy setting
- **COL** — per-column override
- **ADAPT** — derive from the data
- **CFG** — server config / env
- **API** — serve from the backend to the frontend
- **DEL** — remove

### 3.1 Type inference (`profiler.py`)

| Hardcoded | Value | Why it's hardcoded | Problem | Make it |
|---|---|---|---|---|
| Sample used for string inference | first **60** values | PERF: inference runs per column, several times | Sorted / placeholder-prefixed files misread (T3) | ADAPT: seeded random 500 + cache |
| ID name tokens | `id, uuid, key, code, number, no, index…` (English) | MVP: quick, readable heuristic | Floats named `_index` dropped; non-English IDs missed (T2) | ADAPT first (value shape), names as weak prior; COL override |
| Sequential-integer ID rule | unique consecutive ints, > 20 rows | MVP | Year series dropped (T1) | ADAPT: exclude year-like ranges / time names |
| Boolean strings | `yes/no/true/false/y/n/t/f/0/1` | MVP, English | Other languages → categorical | CFG list per locale |
| Year detection | name must contain "year"; range 1800–2100 | CONV + MVP | Unnamed year columns treated as numeric | ADAPT from values; POL range |
| "Numeric but categorical" | ≤ 5 unique and < 5% unique | CONV | 6–25-value codes treated as continuous (T10) | POL + COL override |
| Date detection | ≥ 80% regex hits (50% with name hint), ≥ 75% parse | CONV | Day-first misread (T4) | POL locale / `dayfirst`; COL override |
| Version guard 🔧 | `^v?\d{1,3}(\.\d{1,3}){1,3}$` | SAFE (added this pass) | — | Keep |
| Unit pattern | `$` + number + trailing letters, ≥ 70% | MVP | Currency, separators, negatives, `°` missed (I3, T6) | Shared number/currency parser |
| Multi-label | delimiter in ≥ 40% **and** English name token | MVP | `ingredients` dropped (T7) | ADAPT structure; names as prior |
| Person-name / free text | English name tokens, > 15 unique | MVP | Mostly cosmetic | ADAPT |
| Categorical vs text | ≤ 25 unique or < 5% unique | CONV | — | POL |

### 3.2 Issue detection (`detector.py`)

| Hardcoded | Value | Why | Problem | Make it |
|---|---|---|---|---|
| Non-negative keyword list | `age, price, salary, fare, tenure, weight…` | MVP / DEMO (Titanic "Age") | Unknown names unchecked; `net_income` corrupted (Q6, Q7) | ADAPT (≥ 99% non-negative) + COL min/max rules |
| Missing severity | 5 / 20 / 50 % | CONV | Domain-blind | POL |
| Duplicate severity | 1 / 5 % | CONV | Logs with legit repeats | POL + "duplicates expected" toggle |
| Outlier rule | **1.5 × IQR**, report if > 1% | CONV (Tukey 1977) | Heavy tails over-capped; IQR = 0 skipped (Q5, Q10) | POL: method (IQR k / z / MAD / log-IQR), skew-aware default |
| Constant feature | exactly 1 non-null value | MVP | Near-constant missed (Q4) | POL dominant share |
| Collinearity | Pearson \|r\| > **0.85**, top **5** pairs | CONV + PERF | Non-linear missed; wide data truncated | POL threshold, Spearman option, no cap |
| High cardinality | > **50** values or > 40% unique | PERF (one-hot width) | Predictive codes dropped (Q9) | ADAPT to row count + encoding alternatives |
| Rare grouping | > **10** categories → top 10 | CONV | Ignores coverage | POL frequency share (e.g. < 1%) |
| Imbalance | ratio ≥ 3 (medium), ≥ 5 (high), ≥ 10 or > 90% (critical) | CONV | Multi-class over-flagged (Q14) | POL + entropy-based |

### 3.3 Health score (`scorer.py`)

| Hardcoded | Value | Why | Problem | Make it |
|---|---|---|---|---|
| Weights | missingness 25%, five others 15% each | MVP: one "reasonable" blend | Domain-blind; regression gets 15 free points (M10) | POL presets; re-weight over applicable dimensions |
| Penalty formulas | missing `×2.5 + worst×0.25`, dup `×5`, outliers `avg×3 + n×4`, validity `20/col`, balance `(r−1)×12`, feature `10/15/5` | MVP: hand-tuned on demos | Count-based penalties punish wide datasets; ratio-3 cliff | Normalise by column count; continuous balance; POL coefficients |
| Retention penalty cap | 20 points | SAFE (anti-gaming) | Arbitrary | POL |
| Grade bands | 90/80/70/60 → A–F | UI convention | Five different band schemes across the app | API single definition |
| Summary text | English templates, bands 85/70 | MVP | Language, inconsistent bands | Templates + i18n |

### 3.4 Fix selection (`rules.py`)

| Hardcoded | Value | Why | Problem | Make it |
|---|---|---|---|---|
| One method per issue | median/mean/mode/"Unknown", cap, one-hot, drop… | MVP: deterministic, explainable | No user choice (C2) | POL defaults + per-recommendation alternatives & params |
| Drop column if missing > | **70%** (default off) | CONV | — | POL |
| Mean vs median | \|skew\| > **1** ⇒ median | CONV | Years → fractional (C1) | POL; type-aware |
| Negatives | → median | MVP | No NaN / abs / drop option | Alternatives |
| SMOTE vs weights | ratio ≥ **6** ⇒ SMOTE | CONV | — | POL + alternatives |
| Default approvals | 4 destructive fixes on by default | MVP: "one-click clean" demo flow | Silent data removal (C3) | POL "destructive off" |
| `problem_type` | accepted, never used | MVP leftover | Same rules for both tasks | Use it (e.g. imbalance only for classification) |

### 3.5 Execution & transforms (`executor.py`, `transforms.py`)

| Hardcoded | Value | Why | Problem | Make it |
|---|---|---|---|---|
| Step order | 12 fixed steps | SAFE: avoids order bugs | No flexibility (C4) | Declarative steps with constraints |
| One-hot cap | **50** categories | PERF | Silent skip (C6) | POL + report |
| Multi-hot cap | **15** tokens | PERF | Silent loss | POL + min-frequency |
| Date parts | year, month, day | MVP | Hour / weekday lost (C5) | POL list |
| Date fill fallbacks | 2000 / 6 / 15 | SAFE | Fabricated values | Missing flag instead |
| Numeric fill fallback | 0.0 | SAFE | Fabricated values | NaN + flag |
| Outlier bound fallback | 1st / 99th percentile | SAFE | Hidden second rule | Document / POL |
| Tag delimiters | `, ; \|` | MVP | `/` unsupported (T8) | POL |
| "Other" label | `"Other"` | MVP | English | CFG |

### 3.6 Benchmark (`benchmark.py`)

| Hardcoded | Value | Why | Problem | Make it |
|---|---|---|---|---|
| Row cap | **2,000** | PERF: SVC is O(n²); requests are synchronous | 1–2% of big data (M1, M2) | POL + adaptive time budget, background job |
| Models | RF, LogReg, GB, DT, SVC (+ XGB, LGBM if installed) | MVP | No choice; env-dependent list | POL model subset; registry |
| Hyperparameters | `n_estimators=100`, `max_depth=8/5/3`, `alpha=1`… | LIB + PERF | No tuning | Optional small random search |
| CV folds | ≤ 5, shuffled K-fold | CONV | Time-series leakage (M3) | POL folds + split strategy |
| Metric | macro F1 / R² | MVP | Wrong for fraud / business users | POL metric; compute several |
| Seed | `random_state=42` everywhere | REPRO | No stability view | POL seed; repeated CV |
| Parallelism | `n_jobs=-1` | PERF | Concurrent users fight for cores | CFG worker count |
| Text-column cap | > 50 categories dropped | PERF | Silent | Report |
| Suitability labels | "High" / "Moderate" text per model | UI | Not data-driven | Derive from data size / type |

### 3.7 AI explanations (`explainer.py`, `config.py`)

| Hardcoded | Value | Why | Problem | Make it |
|---|---|---|---|---|
| Provider | Gemini REST only | MVP | Lock-in; free-tier quota exhausted in testing | Provider interface (Gemini / OpenAI / local Ollama) |
| Models | `gemini-3.8-flash` + fallbacks (list repeats primary) | CFG (env-overridable) ✅ | One real fallback | De-duplicate |
| Timeout / retries / budget | 30 s / 2 / 45 s | CFG ✅ | Blocks diagnostics | Async generation |
| Prompt | English, "Senior Data Scientist", 2–3 sentences | MVP | Language / audience fixed | Template file with `{language}`, `{audience}` |
| Fallback texts | ~15 English templates quoting "70%", "1.5× IQR", "top 10" | MVP | **DUP** of thresholds; text lies if config changes | Render thresholds from policy |
| Data sent | `stats_context` incl. sample cell values | MVP | Privacy (S3) | Redaction switch |

### 3.8 Ingest, storage, API (`core/*`, `api/*`)

| Hardcoded | Value | Why | Problem | Make it |
|---|---|---|---|---|
| Upload limit | 200 MB (backend **and** frontend) | SAFE + DUP | Drift → 413 surprises | CFG + API `/config` |
| Extensions | `.csv .xlsx .xls` (backend, `api.ts`, `<input accept>`) | MVP + DUP | Three copies; no Parquet / JSON | Connector registry + API |
| Encodings tried | utf-8-sig, utf-8, cp1252, latin-1 | MVP | latin-1 always "succeeds" → silent mojibake; UTF-16 fails (I8) | Charset detection |
| NA tokens | pandas defaults | LIB | `?`, `-`, `unknown` missed (I4) | POL list |
| dtype inference at read | automatic | LIB | Leading zeros lost (I6) | Read as string first |
| Excel sheet | first non-empty | MVP | README sheets (I9) | Sheet picker / largest |
| Min rows / columns | 10 / 2 | SAFE ✅ | — | CFG |
| Retention | 7 days, startup-only cleanup | CFG ✅ / MVP | Rows linger | Scheduled job |
| CORS | `localhost:3000` | MVP | Deployment | CFG (documented) |
| Ownership | random client ID in `localStorage` | MVP | No accounts / sharing (S1, S2) | Auth |
| Locks | in-process `threading.Lock` | MVP | Multi-worker unsafe (P3) | Redis / DB locks |
| Demo datasets | 3 synthetic generators, fixed seeds | DEMO ✅ | Fine for demos | Optional real public datasets |
| Branding | "AI Data Readiness Platform" (API, PDF, script header) | Legacy name | Inconsistent with "Nebulus" | CFG `PROJECT_NAME` everywhere |

### 3.9 Visual lab (`visualizer.py`, `explore/page.tsx`)

| Hardcoded | Value | Why | Make it |
|---|---|---|---|
| Chart sample | 2,000 rows (≤ 10,000 via API) | PERF: browser rendering + payload size | UI slider |
| Columns sent | 60, quota 15 categorical + 5 dates | PERF | UI "load more columns" |
| Top categories | 15 | PERF | POL |
| PCA fit rows | ≤ 50,000 | PERF | CFG |
| k-means k | 2–6 by silhouette | CONV | UI k slider |
| Correlation columns | 40 least-missing | PERF | UI selection |

### 3.10 Frontend

| Hardcoded | Where | Why | Problem | Make it |
|---|---|---|---|---|
| Health weights "·25% ·15%" | `HealthScoreGauge.tsx` | DUP | Lies if backend changes | API |
| Model names, fake log lines | `Terminal.tsx` | UI (scripted progress) | Shows XGBoost even if not installed | Real progress events (SSE) |
| Pipeline log lines | `RecommendationChecklist.tsx` | UI | 🔧 now filtered to approved fixes; still scripted timing | SSE |
| Target name hints incl. `survived`, `churn`, `medianhousevalue` | `ObjectiveStep.tsx` | DEMO | Demo names in product logic; exact-match only | DEL — use backend suggestions |
| Demo cards (names, sizes, descriptions) | `UploadStep.tsx` | DUP of backend | Drift ("~300 rows") | API |
| Step labels "01/05"…"05/05" | 5 components | MVP | Six edits to add a step | Derive from `STEPS` |
| API / landing URL fallbacks | `api.ts`, `Header.tsx` | Dev convenience | Fine if env set | Keep, document |
| All copy | everywhere | MVP | English only | i18n |
| TS types | `services/api.ts` | DUP of Pydantic | Already drifted | Generate from OpenAPI |

---

## Part 4 — Design for "nothing hardcoded"

The goal isn't "no constants". The goal is **every constant has one home, a documented default, and a way to change it at the right level.**

### 4.1 Four configuration layers

```text
┌───────────────────────────────────────────────────────────────┐
│ 1. System config (env / config.py)                            │  limits, providers, storage, CORS
├───────────────────────────────────────────────────────────────┤
│ 2. Workspace defaults (DB, editable by admin)                 │  default preset, locale, privacy
├───────────────────────────────────────────────────────────────┤
│ 3. Dataset policy (DB JSON per dataset, editable in UI)       │  thresholds, weights, models, metric
├───────────────────────────────────────────────────────────────┤
│ 4. Overrides                                                  │
│    • per column: type, role (id/target/ignore), min/max,      │
│      allowed values, NA tokens, ordinal order                 │
│    • per recommendation: method + params                      │
└───────────────────────────────────────────────────────────────┘
          resolved policy = merge(1 → 2 → 3 → 4), stored with every result
```

### 4.2 Policy schema (sketch)

```python
# backend/app/core/policy.py
class IngestPolicy(BaseModel):
    na_tokens: list[str] = ["", "NA", "N/A", "null", "?", "-", "unknown"]
    decimal: Literal["auto", ".", ","] = "auto"
    dayfirst: Literal["auto", True, False] = "auto"
    keep_leading_zeros: bool = True

class InferencePolicy(BaseModel):
    sample_size: int = 500
    categorical_max_unique: int = 25
    numeric_code_max_unique: int = 5
    year_range: tuple[int, int] = (1800, 2100)

class DetectionPolicy(BaseModel):
    missing_bands: tuple[float, float, float] = (5, 20, 50)
    outlier: Literal["iqr", "log_iqr", "zscore", "mad", "auto"] = "auto"
    outlier_k: float = 1.5
    near_constant_share: float = 0.99
    collinearity: float = 0.85
    collinearity_method: Literal["pearson", "spearman"] = "pearson"
    high_cardinality: Literal["auto"] | int = "auto"
    rare_share: float = 0.01
    leakage_threshold: float = 0.98

class FixPolicy(BaseModel):
    destructive_default_on: bool = False
    numeric_impute: Literal["auto", "mean", "median", "knn", "iterative"] = "auto"
    categorical_encoding: Literal["onehot", "target", "frequency", "auto"] = "auto"
    date_parts: list[str] = ["year", "month", "day", "weekday", "hour"]
    one_hot_max: int = 50

class ScoringPolicy(BaseModel):
    preset: Literal["balanced", "strict", "lenient", "custom"] = "balanced"
    weights: dict[str, float] | None = None       # None → preset
    grade_bands: list[tuple[float, str]] = [(90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "F")]

class BenchmarkPolicy(BaseModel):
    metric: str = "auto"                           # f1_macro / roc_auc / pr_auc / r2 / mae / rmse …
    models: list[str] = ["auto"]
    cv: Literal["kfold", "stratified", "time_series", "group"] = "auto"
    folds: int = 5
    max_rows: int | Literal["auto"] = "auto"
    time_budget_sec: int = 60
    seed: int = 42

class Policy(BaseModel):
    version: int = 1
    ingest: IngestPolicy = IngestPolicy()
    inference: InferencePolicy = InferencePolicy()
    detection: DetectionPolicy = DetectionPolicy()
    fixes: FixPolicy = FixPolicy()
    scoring: ScoringPolicy = ScoringPolicy()
    benchmark: BenchmarkPolicy = BenchmarkPolicy()
```

### 4.3 Registries instead of if/else chains

Detectors, fixes, models, metrics and connectors become plugins:

```python
@detector("near_constant", default_severity="medium")
def near_constant(col: ColumnView, policy: DetectionPolicy) -> Issue | None: ...

@fix("impute_knn", handles=["missing_values"], destructive=False, params={"k": 5})
def impute_knn(df, column, k): ...

@model("lightgbm", tasks=["classification", "regression"], requires="lightgbm")
def lightgbm(task, policy): ...
```

Adding a capability becomes "add one decorated function". The UI lists alternatives straight from the registry, so the frontend never hardcodes method names again.

### 4.4 API additions

| Endpoint | Purpose |
|---|---|
| `GET /config` | Upload limit, extensions, presets, grade bands, available models / metrics / fixes (kills every frontend DUP) |
| `GET / PUT /datasets/{id}/policy` | Read / edit the resolved policy; editing invalidates downstream results (same mechanism as objectives) |
| `PUT /datasets/{id}/columns/{name}` | Column overrides (type, role, rules) |
| `POST /datasets/{id}/recommendations/approve` | Extend from `{id: bool}` to `{id: {approved, method, params}}` |
| `GET /jobs/{id}/events` (SSE) | Real progress for diagnose / execute / benchmark |

### 4.5 Rules for the codebase

1. No numeric literal in engine code except 0, 1 and math constants; everything else comes from `policy`.
2. Explanation text is rendered **from** the policy ("above the {policy.detection.drop_missing}% threshold").
3. The frontend reads limits, labels and lists from `GET /config` and OpenAPI-generated types.
4. Every stored result records the policy version and hash it was produced with, so results are reproducible and comparable.

---

## Part 5 — Features for a one-stop data analysis & EDA platform

Organised by the stages of a real analysis. ★ = already exists (fully or partly).

### 5.1 Connect & ingest

| Feature | Why it matters | Sketch | Effort |
|---|---|---|---|
| ★ CSV / Excel upload | — | — | — |
| Parquet, JSON / JSONL, TSV, Feather | Modern data lands in these | pandas / pyarrow readers in a connector registry | S |
| Databases (Postgres, MySQL, SQLite, BigQuery, Snowflake) | Most business data lives in a database | Connection form + SQL editor; sample or full pull | M |
| Google Sheets, S3 / GCS / Azure Blob, URL | No download-upload dance | OAuth + signed-URL fetchers | M |
| Multi-file workspace + join builder | Real analyses combine tables | Key suggestions by name / overlap; preview join cardinality | L |
| Large-data engine (DuckDB / Polars) | 200 MB+ files, instant profiling | Swap pandas for DuckDB in profiling / visual lab; keep pandas for sklearn | M |

### 5.2 Understand (profiling 2.0)

| Feature | Why | Sketch | Effort |
|---|---|---|---|
| ★ Column profile, types, missing %, stats | — | — | — |
| Column deep-dive drawer | One click → everything about a column | Distribution, top values, patterns, outliers, missing pattern, correlation with target | M |
| Semantic types | Emails, phones, URLs, geo, currency, % | Regex + validators; drives PII and maps | M |
| Pattern profiling | `"AB-1234"` vs `"ab1234"` formats | Mask values (`A`=letter, `9`=digit) and count masks | S |
| Auto-insights feed | "Age has 18% missing, mostly in class 3" | Rule-based + LLM narration of statistically significant findings | M |
| Dataset comparison & drift | Train vs test, this month vs last | PSI / KS per column, schema diff | M |
| Data dictionary | Shared understanding | Editable descriptions, units, owners per column; exported in reports | S |

### 5.3 Explore (visual EDA)

| Feature | Why | Sketch | Effort |
|---|---|---|---|
| ★ Visual lab: distribution, scatter 2D/3D, PCA + clusters, correlation, missingness, group compare, parallel coordinates | — | — | — |
| Target analysis tab | The #1 EDA question: what drives the target? | Feature-vs-target plots, mutual information ranking, churn-rate-by-bin charts | M |
| Pair plot / scatter matrix | Quick multivariate scan | Canvas small multiples with brushing across cells | M |
| Linked brushing across views | Select outliers in one chart → highlighted everywhere | Shared selection store | M |
| Time-series explorer | Trends, seasonality, gaps | Resampling, rolling stats, STL decomposition, ACF/PACF, gap detection | M |
| Geo maps | Lat/long and region data | Point / hex-bin / choropleth (MapLibre) | M |
| Text explorer | Reviews, tickets, comments | Length distribution, top n-grams, word cloud, language detection, sentiment | M |
| Category explorer | Treemap / sunburst for hierarchies | Drill-down by multiple categorical columns | S |
| t-SNE / UMAP embeddings | Non-linear structure | Server-side UMAP on the sample, 2D/3D in the existing scatter engines | S |
| Chart builder | Free-form exploration | Pick chart type + encodings (x, y, color, size, facet) | M |
| Save chart / dashboard | Reuse and share | Pin charts to a dataset dashboard; export PNG / SVG | M |

### 5.4 Test (statistics)

| Feature | Why | Sketch | Effort |
|---|---|---|---|
| Normality tests | Pick the right methods | Shapiro / Anderson + Q-Q plot | S |
| Group comparison tests | "Is the difference real?" | t-test / Mann-Whitney / ANOVA / Kruskal with effect sizes, auto-picked by data shape | S |
| Association tests | Categorical relationships | Chi-square, Cramér's V matrix | S |
| Correlation significance | Avoid reading noise | p-values + confidence intervals on the heatmap | S |
| A/B test analyser | Common business ask | Conversion / mean comparison, power, sample size | M |

### 5.5 Clean & prepare

| Feature | Why | Sketch | Effort |
|---|---|---|---|
| ★ Detected issues + approvable fixes, exported script | — | — | — |
| Method alternatives & parameters per fix | User control (C2) | Registry-driven dropdowns | M |
| Column type / role overrides | Fixes most inference errors | Dropdown in profile table | M |
| Interactive data grid | See and edit rows | Virtualised grid with filter / sort / find-replace; edits recorded as steps | M |
| Value normaliser | Spelling variants (Q1, Q2) | Cluster similar values; merge with one click | S |
| Custom validation rules | Domain knowledge | `age between 0 and 120`, regex, allowed values, cross-column rules | M |
| Recipe history, undo, versioning | Safe experimentation | Each step versioned; diff and revert | M |
| Apply recipe to new data | Monthly reports | Saved recipe + schema-drift check | M |
| Export recipe as notebook / sklearn Pipeline | Hand-off to data scientists | Generate `.ipynb` + fitted `Pipeline` pickle | M |

### 5.6 Engineer features

| Feature | Why | Sketch | Effort |
|---|---|---|---|
| Date part extractor (configurable) | Hour / weekday signals (C5) | Policy list | S |
| Binning (equal-width, quantile, custom) | Interpretable features | UI + transform | S |
| Interactions & ratios | Domain features | Formula builder (`price / area`) | M |
| Encoders: target, frequency, hashing, ordinal | Keep high-cardinality signal (Q9) | Fold-safe target encoding | M |
| Text vectorisation | Use free text instead of dropping it | TF-IDF / small embeddings | M |
| Feature selection | Fewer, better features | Mutual information, RFE, permutation importance | M |

### 5.7 Model & explain

| Feature | Why | Sketch | Effort |
|---|---|---|---|
| ★ Cross-validated leaderboard (classification / regression) | — | — | — |
| Metric picker + several metrics per model | Right metric for the problem (M1) | Policy metric; table shows all | S |
| Baseline, mean ± std, per-fold scores | Honest comparisons (M5, M6) | — | S |
| Hyperparameter search (time-boxed) | Better models | Optuna / random search with a budget | M |
| Explainability | Trust | Permutation importance, SHAP summary, partial dependence | M |
| Confusion matrix, ROC / PR curves, residual plots, calibration | Diagnose models | Charts in the visual lab | M |
| More tasks | Clustering, forecasting, anomaly detection, multi-label | Task registry | L |
| Model export + predict-on-upload | Close the loop | Download a pickle / ONNX; "score a new file" | M |

### 5.8 AI assistant

| Feature | Why | Sketch | Effort |
|---|---|---|---|
| Ask questions in plain language | "What drives churn?", "Show sales by region" | LLM plans → safe, sandboxed pandas / DuckDB queries → chart + answer, with the code shown | L |
| Auto-generated EDA narrative | Executive summary in seconds | Insights feed + LLM write-up, numbers always computed, never invented | M |
| Explain any chart | Teaching tool | "What am I looking at?" button | S |
| Privacy controls | Trust | Schema-only mode (no cell values), local LLM option | S |

### 5.9 Report & share

| Feature | Why | Sketch | Effort |
|---|---|---|---|
| ★ PDF audit, manifest, script, cleaned CSV | — | — | — |
| Full EDA report (HTML + PDF) with charts | One deliverable | Jinja2 template; charts rendered server-side from the visual lab specs | M |
| Jupyter notebook export | Reproduce everything in code | Generated `.ipynb` with profiling, charts, cleaning, modelling cells | M |
| Share links (read-only) | Collaboration | Signed tokens, expiry | S |
| Comments & annotations | Team discussion | Per chart / column / issue | M |

### 5.10 Platform

| Feature | Why | Sketch | Effort |
|---|---|---|---|
| Accounts, projects, roles | Multi-user | OAuth / magic link; project → datasets | M |
| Background jobs + live progress | Big data, no timeouts (P1) | Arq / RQ + SSE | M |
| Object storage, Postgres, Redis | Deployability | Config switches | S–M |
| Public API + Python SDK / CLI | Automation | OpenAPI-generated clients | M |
| i18n | Non-English users | next-intl + backend templates | M |
| Audit log | Governance | Who changed what policy / fix, when | S |

---

## Part 6 — Prioritised roadmap

### Phase 1 — Correctness (1–2 weeks)

Fix everything that silently produces wrong answers.

1. Number normaliser: thousands separators, decimal comma, currency, signed units (I1–I3, T6)
2. NA tokens + sentinel detection; read as string first to keep leading zeros (I4–I6)
3. Type inference: random 500-value sample, year-before-ID, float IDs excluded, day-first detection (T1–T4)
4. Category normaliser for features **and** target (Q1, Q2)
5. Leakage detector (Q3)
6. Data-driven non-negative rule; keywords as hints only (Q6, Q7)
7. Never auto-drop target-correlated high-cardinality features; offer frequency / target encoding (Q9)
8. Scorer: applicable-dimension weighting, continuous balance, column-count normalisation (M10)
9. Benchmark honesty: show "trained on N of M rows", baseline, mean ± std, class weights for all models (M1, M5–M7)

### Phase 2 — Control (2–3 weeks)

Remove the hardcoding.

1. `Policy` object + presets + `GET/PUT /policy` + settings drawer in the UI
2. Column overrides (type, role, rules)
3. Fix alternatives and parameters (registry-driven)
4. `GET /config`; OpenAPI-generated TS types; delete frontend duplicates and demo hints
5. Explanation templates rendered from the policy; privacy switch for the LLM

### Phase 3 — EDA depth (3–4 weeks)

1. Target analysis tab, column deep-dive drawer, auto-insights feed
2. Statistical tests panel
3. Time-series, text and geo explorers
4. Linked brushing, pair plot, UMAP, chart builder, saved dashboards
5. Full EDA report (HTML / PDF with charts) + notebook export

### Phase 4 — Scale & platform (ongoing)

1. Background jobs + SSE progress; DuckDB engine for large files
2. Connectors (Parquet, JSON, databases, cloud storage, Sheets); multi-table joins
3. Accounts, projects, sharing, comments, audit log
4. Metric picker, tuning, explainability, model export, more task types
5. Natural-language assistant with sandboxed execution

---

## Appendix A — Edge-case battery results (after this pass's fixes)

| Case | Result | Observation |
|---|---|---|
| thousands_separator | ❌ | `"1,234"` → text → dropped |
| euro_decimal_comma | ❌ | `"12,50"` → text → dropped |
| currency_symbols | ❌ | `"₹12,345"` → text → dropped |
| disguised_missing_question_mark | ❌ | `"?"` → column text; 0 missing detected |
| sentinel_minus_999 | ❌ | `-999` flagged as invalid negative + outlier, not as missing |
| leading_zero_codes | ❌ | `"00123"` → `123` |
| metadata_rows_above_header | ❌ | Rejected: tokenizing error |
| utf16_csv | ❌ | Rejected: 0 usable columns |
| whitespace_padded_numbers | ✅ | Numeric |
| mixed_type_column | ❌ | Text → dropped as high-cardinality |
| excel_serial_dates | ❌ | Numeric, not date |
| excel_multi_sheet | ❌ | Rejected: used README sheet |
| annual_year_series | ❌ | `Year` → id → dropped |
| continuous_named_index | ❌ | Float `consumer_price_index` → id → dropped |
| first_60_values_placeholder | ❌ | Date column → text |
| dayfirst_dates | ❌ (🔧 partly) | Dates now extracted; 77 / 200 months wrong (month-first) |
| mixed_timezones | 🔧 | Parsed; year / month / day extracted |
| version_strings | 🔧 | No longer treated as dates |
| non_english_multilabel | ❌ | `ingredients` → text → dropped |
| slash_delimited_genres | ❌ | Categorical, not multi-label |
| negative_with_units | ❌ | `"-5 °C"` → text |
| rating_scale_1_to_5 | ✅ | Kept numeric |
| inconsistent_category_spelling | ❌ | 5 variants, no issue |
| target_spelling_variants | ❌ | 4 classes instead of 2 |
| target_leakage_copy | ❌ | No issue; 100% score |
| near_constant_column | ❌ | Not flagged |
| zero_inflated_outliers | ❌ | Not flagged |
| non_negative_unknown_name | ❌ | `mass_kg` negatives not flagged |
| legit_negative_keyword | ❌ | `net_income` "fixed" to median |
| pii_email_phone | ❌ | No PII issue |
| predictive_high_cardinality | ❌ | Predictive zip dropped by default; 53% |
| generated_name_collision_* | 🔧 | Existing columns preserved |
| year_mean_imputation | ❌ | Fractional years |
| ordinal_categories | ❌ | One-hot, order lost |
| rare_fraud_1pct | ❌ | Macro F1 only; 2,000 of 3,000 rows |
| thirty_class_text_target | ✅ | Accepted |
| regression_skewed_target | ❌ | No target-transform suggestion |
| time_ordered_target | 🔧 / ❌ | R² 0.995 now (dates extracted); still random K-fold |
| single_class_after_filter | ✅ | Rejected with clear message |
| wide_800_columns | ✅ | Diagnose 19.7 s, benchmark 20.2 s, visual lab 2.8 s |
| tall_150k_rows | ✅ / ❌ | Fast (≈ 11 s total), but benchmark on 2,000 of 150,000 rows |

## Appendix B — Related documents

- `ARCHITECTURE.md`: how the system works today
- `HARDCODED_AUDIT.md`: line-level inventory of hardcoded values
- `backend/tests/edge_case_battery.py`: reproduces every observation above
