"""
Edge-case battery: runs ~40 tricky datasets through the same flow as the API (ingest -> objective ->
diagnose -> execute -> benchmark -> visual lab), in-process and without an LLM, and prints findings.

    cd backend
    venv\\Scripts\\python tests\\edge_case_battery.py [case_name ...]
"""
import io
import os
import sys
import tempfile
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.api.v1.endpoints.objectives import validate_target
from app.core.storage import DatasetParseError, parse_uploaded_file, read_dataset
from app.engine.benchmark import BenchmarkError, benchmark_models
from app.engine.detector import detect_issues
from app.engine.executor import execute_pipeline
from app.engine.profiler import profile_dataset
from app.engine.rules import select_remediation_methods
from app.engine.scorer import calculate_health_score
from app.engine.visualizer import build_explore_payload

rng = np.random.default_rng(3)
TMP = tempfile.mkdtemp()


def csv(df, **kw):
    return df.to_csv(index=False, **kw).encode(kw.pop("encoding", "utf-8") if "encoding" in kw else "utf-8")


def base(n=200):
    return pd.DataFrame({"f1": rng.normal(0, 1, n).round(3), "f2": rng.normal(5, 2, n).round(3), "y": rng.choice(["a", "b"], n)})


# Each case: name -> (bytes, ext, target, problem_type, checker(ctx) -> list[str] of observations)
CASES = {}


def case(name, target="y", ptype="classification", ext=".csv"):
    def wrap(fn):
        CASES[name] = (fn, target, ptype, ext)
        return fn
    return wrap


def T(ctx, col):
    return ctx["types"].get(col, "<absent>")


# ─── Ingest & parsing ────────────────────────────────────────────────────────

@case("thousands_separator")
def _():
    d = base(); d["revenue"] = [f"{v:,}" for v in rng.integers(1000, 900000, len(d))]
    return csv(d), lambda c: [f"'1,234' strings typed as {T(c,'revenue')}", f"revenue survives cleaning: {'revenue' in c['cleaned'].columns}"]


@case("euro_decimal_comma")
def _():
    d = base(); d["price"] = [f"{v:.2f}".replace(".", ",") for v in rng.uniform(1, 999, len(d))]
    return d.to_csv(index=False, sep=";").encode(), lambda c: [f"'12,50' typed as {T(c,'price')}", f"price survives: {'price' in c['cleaned'].columns}"]


@case("currency_symbols")
def _():
    d = base(); d["cost"] = [f"₹{v:,}" for v in rng.integers(100, 99999, len(d))]
    return csv(d), lambda c: [f"'₹12,345' typed as {T(c,'cost')}", f"cost survives: {'cost' in c['cleaned'].columns}"]


@case("disguised_missing_question_mark")
def _():
    d = base(); h = rng.integers(1, 80, len(d)).astype(object); h[rng.choice(len(d), 30, replace=False)] = "?"; d["hours"] = h
    return csv(d), lambda c: [f"'?' placeholders → column typed {T(c,'hours')}",
                              f"missing detected in hours: {any(i['issue_type']=='missing_values' and i.get('column')=='hours' for i in c['issues'])}"]


@case("sentinel_minus_999")
def _():
    d = base(); inc = rng.normal(50000, 9000, len(d)).round(0); inc[rng.choice(len(d), 20, replace=False)] = -999; d["income"] = inc
    return csv(d), lambda c: [f"-999 sentinel treated as missing: {any(i['issue_type']=='missing_values' and i.get('column')=='income' for i in c['issues'])}",
                              f"issues on income: {[i['issue_type'] for i in c['issues'] if i.get('column')=='income']}"]


@case("leading_zero_codes")
def _():
    d = base(); d["zip"] = [f"{v:05d}" for v in rng.integers(1, 999, len(d))]
    return csv(d), lambda c: [f"'00123' after storage → {c['stored']['zip'].head(2).tolist()} (dtype {c['stored']['zip'].dtype})", f"typed {T(c,'zip')}"]


@case("metadata_rows_above_header")
def _():
    body = csv(base()).decode()
    return ("Sales report generated 2024-01-01\nConfidential\n\n" + body).encode(), lambda c: [f"parsed columns: {list(c['stored'].columns)[:4]}"]


@case("utf16_csv")
def _():
    return base().to_csv(index=False).encode("utf-16"), lambda c: [f"parsed columns: {list(c['stored'].columns)[:4]}"]


@case("whitespace_padded_numbers")
def _():
    d = base(); d["qty"] = [f"  {v} " for v in rng.integers(1, 500, len(d))]
    return csv(d), lambda c: [f"'  12 ' typed as {T(c,'qty')}"]


@case("mixed_type_column")
def _():
    d = base(); v = rng.integers(1, 100, len(d)).astype(object); v[::7] = "twelve"; d["weight"] = v
    return csv(d), lambda c: [f"numbers + words typed as {T(c,'weight')}; issues: {[i['issue_type'] for i in c['issues'] if i.get('column')=='weight']}"]


@case("excel_serial_dates", ext=".xlsx")
def _():
    d = base(); d["OrderDate"] = rng.integers(43000, 45500, len(d))
    buf = io.BytesIO(); d.to_excel(buf, index=False)
    return buf.getvalue(), lambda c: [f"Excel serial dates (44927) typed as {T(c,'OrderDate')}"]


@case("excel_multi_sheet", ext=".xlsx")
def _():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        pd.DataFrame({"note": ["readme"] * 3}).to_excel(w, sheet_name="README", index=False)
        base().to_excel(w, sheet_name="data", index=False)
    return buf.getvalue(), lambda c: [f"sheet used → columns {list(c['stored'].columns)}", f"warnings: {c['warnings']}"]


# ─── Type inference ──────────────────────────────────────────────────────────

@case("annual_year_series", target="gdp", ptype="regression")
def _():
    n = 34
    d = pd.DataFrame({"Year": np.arange(1990, 1990 + n), "pop": rng.normal(50, 5, n).round(2), "gdp": rng.normal(100, 20, n).round(2)})
    return csv(d), lambda c: [f"Year typed as {T(c,'Year')}", f"Year survives cleaning: {'Year' in c['cleaned'].columns}"]


@case("continuous_named_index")
def _():
    d = base(); d["consumer_price_index"] = rng.uniform(90, 130, len(d)).round(6)
    return csv(d), lambda c: [f"unique float 'consumer_price_index' typed {T(c,'consumer_price_index')}", f"survives: {'consumer_price_index' in c['cleaned'].columns}"]


@case("first_60_values_placeholder")
def _():
    n = 200; d = base(n)
    dates = pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d").tolist()
    d["signup_date"] = ["pending"] * 70 + dates[70:]
    return csv(d), lambda c: [f"date column with 70 leading 'pending' typed {T(c,'signup_date')}"]


@case("dayfirst_dates")
def _():
    n = 200; d = base(n)
    days = pd.date_range("2024-01-01", periods=n)
    d["visit"] = days.strftime("%d/%m/%Y")
    def chk(c):
        if "visit_month" not in c["cleaned"].columns:
            return [f"visit typed {T(c,'visit')}; no month extracted"]
        wrong = int((c["cleaned"]["visit_month"].to_numpy() != days.month.to_numpy()).sum())
        return [f"dd/mm/yyyy dates: {wrong}/{n} rows got the wrong month"]
    return csv(d), chk


@case("mixed_timezones")
def _():
    n = 200; d = base(n)
    d["ts"] = [f"2024-03-{(i%28)+1:02d}T10:00:00{'+05:30' if i%2 else 'Z'}" for i in range(n)]
    return csv(d), lambda c: [f"ts typed {T(c,'ts')}; extracted: {[x for x in c['cleaned'].columns if x.startswith('ts')]}"]


@case("version_strings")
def _():
    d = base(); d["app_version"] = [f"{rng.integers(1,4)}.{rng.integers(0,12)}.{rng.integers(0,9)}" for _ in range(len(d))]
    return csv(d), lambda c: [f"'1.2.3' typed {T(c,'app_version')}"]


@case("non_english_multilabel")
def _():
    toks = ["salt", "sugar", "flour", "egg", "milk", "butter", "yeast"]
    d = base(); d["ingredients"] = [", ".join(rng.choice(toks, rng.integers(1, 4), replace=False)) for _ in range(len(d))]
    return csv(d), lambda c: [f"comma lists named 'ingredients' typed {T(c,'ingredients')}", f"survives as features: {[x for x in c['cleaned'].columns if x.startswith('ingredients')][:3]}"]


@case("slash_delimited_genres")
def _():
    toks = ["Action", "Drama", "Comedy", "Horror", "Romance"]
    d = base(); d["genres"] = ["/".join(rng.choice(toks, rng.integers(1, 3), replace=False)) for _ in range(len(d))]
    return csv(d), lambda c: [f"'Action/Drama' named genres typed {T(c,'genres')}"]


@case("negative_with_units")
def _():
    d = base(); d["temp"] = [f"{v} °C" for v in rng.integers(-20, 40, len(d))]
    return csv(d), lambda c: [f"'-5 °C' typed {T(c,'temp')}"]


@case("rating_scale_1_to_5")
def _():
    d = base(); d["rating"] = rng.integers(1, 6, len(d))
    return csv(d), lambda c: [f"1–5 rating typed {T(c,'rating')} (one-hot → order lost: {any(x.startswith('rating_') for x in c['cleaned'].columns)})"]


# ─── Semantics & data quality detection ──────────────────────────────────────

@case("inconsistent_category_spelling")
def _():
    d = base(); d["city"] = rng.choice(["Delhi", "delhi ", "DELHI", "Mumbai", "mumbai"], len(d))
    return csv(d), lambda c: [f"categories seen: {sorted(c['stored']['city'].unique().tolist())}",
                              f"inconsistency issue raised: {any('city' == i.get('column') and 'incons' in i['issue_type'] for i in c['issues'])}"]


@case("target_spelling_variants")
def _():
    d = base(); d["y"] = rng.choice(["Yes", "yes ", "No", "no"], len(d))
    return csv(d), lambda c: [f"target classes: {sorted(c['stored']['y'].unique().tolist())} → treated as {c['stored']['y'].nunique()} classes"]


@case("target_leakage_copy")
def _():
    d = base(); d["y_encoded"] = (d["y"] == "a").astype(int)
    return csv(d), lambda c: [f"leakage issue raised: {any('leak' in i['issue_type'] for i in c['issues'])}",
                              f"best model score: {c['best']}"]


@case("near_constant_column")
def _():
    d = base(400); v = np.zeros(400); v[:2] = 1; d["flag"] = v
    return csv(d), lambda c: [f"99.5% one value flagged: {any(i.get('column')=='flag' and i['issue_type']=='constant_feature' for i in c['issues'])}"]


@case("zero_inflated_outliers")
def _():
    d = base(400); v = np.zeros(400); v[:40] = rng.integers(1, 4, 40); v[40:45] = [80, 95, 120, 150, 200]; d["num_claims"] = v
    return csv(d), lambda c: [f"extreme values in 90%-zero column flagged: {any(i.get('column')=='num_claims' and i['issue_type']=='statistical_outliers' for i in c['issues'])}"]


@case("non_negative_unknown_name")
def _():
    d = base(); m = rng.uniform(1, 90, len(d)); m[:6] = -5; d["mass_kg"] = m
    return csv(d), lambda c: [f"negatives in 'mass_kg' flagged: {any(i.get('column')=='mass_kg' and i['issue_type']=='invalid_domain_values' for i in c['issues'])}"]


@case("legit_negative_keyword")
def _():
    d = base(); d["net_income"] = rng.normal(0, 5000, len(d)).round(2)
    return csv(d), lambda c: [f"legitimately negative 'net_income' 'fixed' to median: {any(i.get('column')=='net_income' and i['issue_type']=='invalid_domain_values' for i in c['issues'])}"]


@case("pii_email_phone")
def _():
    d = base(); d["email"] = [f"user{i}@mail.com" for i in range(len(d))]; d["phone"] = [f"+91 98{rng.integers(10000000, 99999999)}" for _ in range(len(d))]
    return csv(d), lambda c: [f"PII issue raised: {any('pii' in i['issue_type'] for i in c['issues'])}; email typed {T(c,'email')}"]


@case("predictive_high_cardinality")
def _():
    n = 600; zips = [f"Z{i:03d}" for i in range(80)]; z = rng.choice(zips, n)
    y = np.array(["a" if int(s[1:]) < 40 else "b" for s in z])
    d = pd.DataFrame({"f1": rng.normal(0, 1, n), "zip_code": z, "y": y})
    return csv(d), lambda c: [f"zip_code (80 values, perfectly predictive) dropped by default: {'zip_code' not in c['cleaned'].columns and not any(x.startswith('zip_code') for x in c['cleaned'].columns)}",
                              f"best model score: {c['best']}"]


@case("generated_name_collision_flag")
def _():
    d = base(); cab = rng.choice(["C1", "C2", None], len(d)).astype(object); d["Cabin"] = cab; d["has_Cabin"] = rng.choice([5.0, 7.0], len(d))
    return csv(d), lambda c: [f"pre-existing 'has_Cabin' values after cleaning: {sorted(c['cleaned']['has_Cabin'].unique().tolist())[:4] if 'has_Cabin' in c['cleaned'] else 'absent'} (original 5.0/7.0)"]


@case("generated_name_collision_date")
def _():
    d = base(); d["Date"] = pd.date_range("2020-01-01", periods=len(d)).astype(str); d["Date_year"] = 1.0
    return csv(d), lambda c: [f"pre-existing 'Date_year'=1.0 after cleaning: {sorted(c['cleaned']['Date_year'].unique().tolist())[:3] if 'Date_year' in c['cleaned'] else 'absent'}"]


@case("year_mean_imputation", target="price", ptype="regression")
def _():
    n = 200; yb = rng.integers(1950, 2020, n).astype(float); yb[:30] = np.nan
    d = pd.DataFrame({"YearBuilt": yb, "area": rng.uniform(50, 300, n).round(1), "price": rng.normal(300, 60, n).round(1)})
    return csv(d), lambda c: [f"imputed YearBuilt values non-integer: {bool((c['cleaned']['YearBuilt'] % 1 != 0).any()) if 'YearBuilt' in c['cleaned'] else 'absent'}"]


@case("ordinal_categories")
def _():
    d = base(); d["level"] = rng.choice(["Low", "Medium", "High"], len(d))
    return csv(d), lambda c: [f"ordinal Low/Medium/High encoded as: {[x for x in c['cleaned'].columns if x.startswith('level')]}"]


# ─── Targets & modelling ─────────────────────────────────────────────────────

@case("rare_fraud_1pct")
def _():
    n = 3000; d = pd.DataFrame({"amt": rng.gamma(2, 50, n), "hour": rng.integers(0, 24, n), "y": np.where(rng.random(n) < 0.01, "fraud", "ok")})
    return csv(d), lambda c: [f"metric: {c['metric']}; best {c['best']}; trained rows capped: {c['rows_note']}"]


@case("thirty_class_text_target")
def _():
    n = 300; d = base(n); d["y"] = rng.choice([f"class_{i}" for i in range(30)], n)
    return csv(d), lambda c: [f"objective accepted; best {c['best']}"]


@case("regression_skewed_target", target="sales", ptype="regression")
def _():
    n = 500; x = rng.normal(0, 1, n); d = pd.DataFrame({"x": x, "z": rng.normal(0, 1, n), "sales": np.exp(2 + x + rng.normal(0, .5, n)).round(2)})
    return csv(d), lambda c: [f"log-transform suggested for skewed target: {any('log' in r['method'].lower() for r in c['recs'])}; best {c['best']}"]


@case("time_ordered_target", target="value", ptype="regression")
def _():
    n = 500; t = np.arange(n); d = pd.DataFrame({"t": t, "noise": rng.normal(0, 1, n), "value": (t * 0.5 + rng.normal(0, 5, n)).round(2),
                                                  "date": pd.date_range("2022-01-01", periods=n).astype(str)})
    return csv(d), lambda c: [f"time-aware CV used: False (random KFold); best {c['best']}"]


@case("single_class_after_filter")
def _():
    d = base(); d["y"] = "a"; d.loc[0, "y"] = "b"
    return csv(d), lambda c: ["(should be rejected at objective)"]


# ─── Scale ───────────────────────────────────────────────────────────────────

@case("wide_800_columns")
def _():
    n = 150; d = pd.DataFrame({f"c{i}": rng.normal(0, 1, n).round(3) for i in range(800)}); d["y"] = rng.choice(["a", "b"], n)
    return csv(d), lambda c: [f"timings: {c['timings']}"]


@case("tall_150k_rows")
def _():
    n = 150_000
    d = pd.DataFrame({"a": rng.normal(0, 1, n).round(3), "b": rng.integers(0, 100, n), "cat": rng.choice(list("abcdefghij"), n),
                      "when": pd.date_range("2000-01-01", periods=n, freq="h").astype(str), "y": rng.choice(["a", "b"], n)})
    return csv(d), lambda c: [f"timings: {c['timings']}; trained rows capped: {c['rows_note']}"]


def run(name, fn, target, ptype, ext):
    content, checker = fn()
    ctx = {"timings": {}}
    t0 = time.time()
    try:
        df, warns = parse_uploaded_file(content, ext)
    except DatasetParseError as e:
        return "REJECTED@ingest", [str(e)[:140]]
    ctx["warnings"] = warns
    path = os.path.join(TMP, f"{name}.csv"); df.to_csv(path, index=False)
    stored = read_dataset(path); ctx["stored"] = stored
    ctx["timings"]["ingest"] = round(time.time() - t0, 1)
    if target not in stored.columns:
        return "TARGET-MISSING", [f"columns parsed: {list(stored.columns)[:5]}"] + checker_safe(checker, ctx)
    try:
        validate_target(stored, target, ptype)
    except HTTPException as e:
        return "REJECTED@objective", [e.detail[:140]]
    t1 = time.time()
    summary, profiles = profile_dataset(stored, target_col=target)
    issues = detect_issues(stored, target_col=target, problem_type=ptype)
    score, subs, grade, _, _ = calculate_health_score(summary, profiles, issues, problem_type=ptype)
    recs = select_remediation_methods(issues, profiles, problem_type=ptype)
    ctx["timings"]["diagnose"] = round(time.time() - t1, 1)
    ctx.update(types={p["name"]: p["inferred_type"] for p in profiles}, issues=issues, recs=recs, score=score)
    t2 = time.time()
    rec_dicts = [{"issue_type": r["issue_type"], "column": r.get("column"), "method": r["method"], "params": r["stats_context"],
                  "is_approved": r["is_approved"]} for r in recs]
    result = execute_pipeline(stored, rec_dicts, target_col=target, problem_type=ptype)
    cpath = os.path.join(TMP, f"{name}_clean.csv"); result.df_model_ready.to_csv(cpath, index=False)
    ctx["cleaned"] = read_dataset(cpath)
    ctx["timings"]["execute"] = round(time.time() - t2, 1)
    t3 = time.time()
    strategy = next((r["stats_context"]["method_key"] for r in recs if r["issue_type"] == "class_imbalance" and r["is_approved"]), None)
    try:
        lb, metric, _, folds, _evaluation = benchmark_models(ctx["cleaned"], target, ptype, strategy)
        ctx["best"] = f"{lb[0]['model_name']} {lb[0]['metric_display']}" if lb else "none"
        ctx["metric"] = metric
        ctx["rows_note"] = f"{min(len(ctx['cleaned']), 2000)} of {len(ctx['cleaned'])}"
    except BenchmarkError as e:
        ctx["best"] = f"BENCHMARK REFUSED: {e}"
        ctx["metric"] = "-"; ctx["rows_note"] = "-"
    ctx["timings"]["benchmark"] = round(time.time() - t3, 1)
    t4 = time.time()
    build_explore_payload(stored, target=target)
    ctx["timings"]["explore"] = round(time.time() - t4, 1)
    return f"OK score={score}", checker_safe(checker, ctx)


def checker_safe(checker, ctx):
    try:
        return checker(ctx)
    except Exception as e:
        return [f"checker error: {type(e).__name__}: {e}"]


only = sys.argv[1:] or list(CASES)
for name in only:
    fn, target, ptype, ext = CASES[name]
    try:
        status, notes = run(name, fn, target, ptype, ext)
    except Exception as e:
        status, notes = "CRASH", [f"{type(e).__name__}: {str(e)[:200]}", traceback.format_exc().splitlines()[-3].strip()]
    print(f"\n## {name} - {status}")
    for n_ in notes:
        print(f"   - {n_}")
    sys.stdout.flush()
