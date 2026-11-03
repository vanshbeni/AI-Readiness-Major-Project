import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from app.engine.profiler import infer_column_type

KIND_BY_TYPE = {
    "numeric": "numeric",
    "year_range": "numeric",
    "boolean": "categorical",
    "categorical": "categorical",
    "datetime": "datetime",
    "id": "id",
}
KIND_QUOTA = {"categorical": 15, "datetime": 5}
TOP_CATEGORIES = 15
TEXT_PREVIEW_CHARS = 60
FIT_ROW_CAP = 50_000


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _datetime(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce", format="mixed")


def _round(values: np.ndarray, digits: int = 5) -> List[Optional[float]]:
    return [None if np.isnan(v) else round(float(v), digits) for v in values]


def _numeric_stats(values: pd.Series) -> Optional[Dict[str, float]]:
    clean = values.dropna()
    if clean.empty:
        return None
    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()) if len(clean) > 1 else 0.0,
        "q1": float(clean.quantile(0.25)),
        "q3": float(clean.quantile(0.75)),
        "skew": float(clean.skew()) if len(clean) > 2 else 0.0,
    }


def _projection(full: pd.DataFrame, sample: pd.DataFrame, features: List[str]) -> Optional[Dict[str, Any]]:
    """PCA of the numeric features (fit on up to FIT_ROW_CAP rows) plus k-means clusters on the sample."""
    usable = [c for c in features if _numeric(full[c]).notna().sum() > 1 and _numeric(full[c]).nunique() > 1]
    if len(usable) < 2 or len(sample) < 10:
        return None

    fit_rows = full if len(full) <= FIT_ROW_CAP else full.sample(FIT_ROW_CAP, random_state=42)
    X_fit = np.column_stack([_numeric(fit_rows[c]).to_numpy(dtype=float) for c in usable])
    X_s = np.column_stack([_numeric(sample[c]).to_numpy(dtype=float) for c in usable])

    imputer = SimpleImputer(strategy="median").fit(X_fit)
    scaler = StandardScaler().fit(imputer.transform(X_fit))
    Z_fit = scaler.transform(imputer.transform(X_fit))
    Z_s = scaler.transform(imputer.transform(X_s))

    n_comp = min(3, len(usable))
    pca = PCA(n_components=n_comp, random_state=42).fit(Z_fit)
    coords = pca.transform(Z_s)

    loadings = []
    for comp in pca.components_:
        order = np.argsort(-np.abs(comp))[:4]
        loadings.append([{"feature": usable[i], "weight": round(float(comp[i]), 3)} for i in order])

    clusters, best_k, best_score = None, None, -1.0
    cluster_space = Z_s[:, : min(len(usable), 8)] if len(usable) > 8 else Z_s
    if len(sample) >= 30:
        for k in range(2, 7):
            if k >= len(sample):
                break
            labels = KMeans(n_clusters=k, n_init=4, random_state=42).fit_predict(cluster_space)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(cluster_space, labels, sample_size=min(len(sample), 1500), random_state=42)
            if score > best_score:
                best_k, best_score, clusters = k, float(score), labels
    return {
        "features": usable,
        "dims": n_comp,
        "explained": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "loadings": loadings,
        "coords": [[round(float(v), 4) for v in row] for row in coords],
        "clusters": [f"cluster {int(c) + 1}" for c in clusters] if clusters is not None else None,
        "k": best_k,
        "silhouette": round(best_score, 3) if clusters is not None else None,
    }


def build_explore_payload(
    df: pd.DataFrame,
    target: Optional[str] = None,
    max_rows: int = 2000,
    max_cols: int = 60,
) -> Dict[str, Any]:
    """
    Builds a column-oriented, JSON-ready view of a dataset for interactive charts:
    typed per-column samples, full-data stats, correlation matrices and a 2D/3D projection.
    """
    total_rows = len(df)
    sample = df.sample(max_rows, random_state=42).sort_index() if total_rows > max_rows else df

    columns: List[Dict[str, Any]] = []
    for col in df.columns:
        col = str(col)
        inferred = infer_column_type(df[col], col)
        kind = KIND_BY_TYPE.get(inferred, "text")
        series = df[col]
        info: Dict[str, Any] = {
            "name": col,
            "kind": kind,
            "inferred_type": inferred,
            "is_target": col == target,
            "missing_pct": round(float(series.isna().mean() * 100), 2) if total_rows else 0.0,
            "unique": int(series.nunique()),
        }
        if kind == "numeric":
            info["stats"] = _numeric_stats(_numeric(series))
        elif kind in ("categorical", "text", "id"):
            counts = series.astype("string").value_counts().head(TOP_CATEGORIES)
            info["top"] = [{"value": str(v), "count": int(n)} for v, n in counts.items()]
        elif kind == "datetime":
            parsed = _datetime(series).dropna()
            if not parsed.empty:
                info["range"] = [parsed.min().isoformat(), parsed.max().isoformat()]
        columns.append(info)

    # Wide datasets: keep the target, a quota of categorical/date columns, then fill with numeric,
    # then whatever is left (text, ids), so no column kind is crowded out entirely.
    order: List[str] = [c["name"] for c in columns if c["is_target"]]
    for kind, quota in KIND_QUOTA.items():
        order += [c["name"] for c in columns if c["kind"] == kind and c["name"] not in order][:quota]
    for kind in ("numeric", "categorical", "datetime", "text", "id"):
        order += [c["name"] for c in columns if c["kind"] == kind and c["name"] not in order]
    included = set(order[:max_cols])
    for c in columns:
        c["included"] = c["name"] in included
        c["numeric_like"] = c["kind"] == "numeric" or (
            c["kind"] == "categorical" and pd.api.types.is_numeric_dtype(df[c["name"]])
        )

    data: Dict[str, List[Any]] = {}
    for c in columns:
        if not c["included"]:
            continue
        s = sample[c["name"]]
        if c["kind"] == "numeric":
            data[c["name"]] = _round(_numeric(s).to_numpy(dtype=float))
        elif c["kind"] == "datetime":
            parsed = _datetime(s)
            data[c["name"]] = [None if pd.isna(v) else int(v.value // 1_000_000) for v in parsed]
        else:
            data[c["name"]] = [
                None if pd.isna(v) else str(v)[:TEXT_PREVIEW_CHARS]
                for v in s.astype(object)
            ]

    numeric_cols = [c["name"] for c in columns if c["numeric_like"] and c["unique"] > 1]
    numeric_cols = sorted(numeric_cols, key=lambda n: df[n].isna().mean())[:40]
    correlation = None
    if len(numeric_cols) >= 2:
        num = pd.DataFrame({n: _numeric(df[n]) for n in numeric_cols})
        pearson = num.corr(method="pearson").round(4)
        spearman = num.corr(method="spearman").round(4)
        correlation = {
            "columns": numeric_cols,
            "pearson": [_round(row, 4) for row in pearson.to_numpy(dtype=float)],
            "spearman": [_round(row, 4) for row in spearman.to_numpy(dtype=float)],
        }

    features = [n for n in numeric_cols if n != target]
    projection = _projection(df, sample, features)

    return {
        "total_rows": total_rows,
        "sample_rows": len(sample),
        "total_cols": len(columns),
        "target": target,
        "columns": columns,
        "data": data,
        "correlation": correlation,
        "projection": projection,
    }
