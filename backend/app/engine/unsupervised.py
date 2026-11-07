"""
Unsupervised tasks: clustering and anomaly detection.

There is no target to score against, so quality is judged against a reference that has NO structure:
  * clustering: the same data with every column shuffled independently keeps each column's distribution but destroys
    the relationships between columns. Real groups must separate clearly better than that (silhouette lift).
  * anomaly detection: several detectors with different assumptions vote; a method is trusted when it agrees with
    the others (leave-one-out consensus), and the reference is flagging the same number of rows at random.
Every row of the cleaned data gets an assignment (cluster id, or anomaly flag + score) for export.
"""
import logging
import re
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from app.engine.benchmark import BenchmarkError, _prepare_features

logger = logging.getLogger(__name__)

MAX_FIT_ROWS = 3000
SILHOUETTE_SAMPLE = 2000
MAX_K = 10
STRUCTURE_ROWS = 1500
MIN_CLUSTER_SHARE = 0.03
# Clustering verdict: silhouette lift over the shuffled-columns reference
USEFUL_GAP, WEAK_GAP, USEFUL_SILHOUETTE = 0.10, 0.03, 0.20
# Anomaly verdict: leave-one-out agreement with the other detectors (Jaccard)
USEFUL_AGREEMENT, WEAK_AGREEMENT = 0.5, 0.25


# ─── Shared preparation ──────────────────────────────────────────────────────

BOOKKEEPING_FLAG = re.compile(r"_was_(missing|capped)(_\d+)?$")


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric feature frame (one-hot for remaining text, median-imputed, constant columns removed). Flags written by
    cleaning ('<col>_was_missing', '<col>_was_capped') describe the data's history, not the rows, so they are left out:
    otherwise 'rows whose value was filled' would become a cluster or an anomaly pattern.
    """
    df = df[[c for c in df.columns if not BOOKKEEPING_FLAG.search(str(c))]]
    X = _prepare_features(df)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns, index=df.index)
    X = X.loc[:, X.std() > 0]
    if X.shape[1] == 0:
        raise BenchmarkError("No varying feature columns remain. Unsupervised learning needs at least one.")
    return X


def _scaled(X: pd.DataFrame) -> np.ndarray:
    """Continuous columns standardised; 0/1 indicators kept as 0/1. Standardising a rare flag (5% ones) would turn
    it into a ±4σ jump that outweighs every real measurement and splits clusters on it."""
    Z = X.to_numpy(dtype=float).copy()
    binary = np.array([set(np.unique(Z[:, j])) <= {0.0, 1.0} for j in range(Z.shape[1])])
    if (~binary).any():
        Z[:, ~binary] = StandardScaler().fit_transform(Z[:, ~binary])
    return Z


def _sample_index(n: int, limit: int, seed: int = 42) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))


def column_groups(X: pd.DataFrame, sources: Optional[Dict[str, str]] = None, originals: Optional[List[str]] = None) -> List[List[int]]:
    """
    Indices of columns that must be shuffled together because they come from the same original column: one-hot
    indicators, date parts (year / month / day), value + unit splits. Shuffling them separately would create rows
    no real data could have (two categories at once, impossible dates) and fake 'structure' in the comparison.
    The source comes from cleaning lineage when known, else from the longest original column name that prefixes it.
    """
    cols = [str(c) for c in X.columns]
    names = sorted(originals or [], key=len, reverse=True)

    def source(c: str) -> str:
        if sources and sources.get(c):
            return sources[c]
        for o in names:
            if c == o or c.startswith(f"{o}_"):
                return o
        return c

    by_source: Dict[str, List[int]] = {}
    for i, c in enumerate(cols):
        by_source.setdefault(source(c), []).append(i)
    return list(by_source.values())


def _shuffle_columns(Z: np.ndarray, groups: Optional[List[List[int]]] = None, seed: int = 7) -> np.ndarray:
    """Same per-column distributions, no relationships between columns (one-hot groups move as one)."""
    rng = np.random.default_rng(seed)
    out = Z.copy()
    for g in groups or [[j] for j in range(Z.shape[1])]:
        out[:, g] = Z[rng.permutation(len(Z))][:, g]
    return out


def _without_tiny(labels: np.ndarray) -> np.ndarray:
    """Clusters smaller than MIN_CLUSTER_SHARE of the rows are a handful of outliers split off, not a segment:
    they count as noise (-1), so silhouette cannot reward 'everyone vs the 6 extreme rows'."""
    labels = np.asarray(labels).copy()
    floor = max(5, int(np.ceil(MIN_CLUSTER_SHARE * len(labels))))
    for cl, size in zip(*np.unique(labels[labels >= 0], return_counts=True)):
        if size < floor:
            labels[labels == cl] = -1
    return labels


def _silhouette(Z: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Silhouette over real clusters, discounted by the share of rows left as noise / tiny clusters."""
    labels = _without_tiny(labels)
    mask = labels >= 0
    if mask.sum() < 10 or len(set(labels[mask])) < 2:
        return None
    Zm, lm = Z[mask], labels[mask]
    size = min(SILHOUETTE_SAMPLE, len(lm))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = float(silhouette_score(Zm, lm, sample_size=size, random_state=42))
    return s * float(mask.mean())


def _k_range(n: int) -> List[int]:
    return list(range(2, max(3, min(MAX_K, n // 10) + 1)))


def _best_kmeans_silhouette(Z: np.ndarray, ks: List[int]) -> Tuple[float, int]:
    best, best_k = -1.0, ks[0]
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=4, random_state=42).fit_predict(Z)
        s = _silhouette(Z, labels)
        if s is not None and s > best:
            best, best_k = s, k
    return best, best_k


def cluster_structure(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Does the data contain groups at all? Best k-means silhouette on the real data vs the same data with each column
    shuffled independently (same distributions, no relationships). Returns None when it cannot be assessed.
    """
    try:
        X = feature_frame(df)
    except BenchmarkError:
        return None
    if len(X) < 30:
        return None
    idx = _sample_index(len(X), STRUCTURE_ROWS)
    Z = _scaled(X.iloc[idx])
    ks = list(range(2, min(7, len(idx) // 10 + 1))) or [2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        real, k = _best_kmeans_silhouette(Z, ks)
        null, _ = _best_kmeans_silhouette(_shuffle_columns(Z, column_groups(X, originals=[str(c) for c in df.columns])), ks)
    return {"silhouette": round(real, 4), "null_silhouette": round(null, 4), "gap": round(real - null, 4), "k": k,
            "rows": int(len(idx)), "features": int(X.shape[1])}


def _assign_all(Z_all: np.ndarray, Z_fit: np.ndarray, labels_fit: np.ndarray, model: Any) -> np.ndarray:
    """Labels for every row: the model's own predict when it has one, else nearest fitted neighbours."""
    if hasattr(model, "predict") and not isinstance(model, (AgglomerativeClustering, DBSCAN)):
        try:
            return np.asarray(model.predict(Z_all))
        except Exception:
            pass
    knn = KNeighborsClassifier(n_neighbors=min(5, len(Z_fit))).fit(Z_fit, labels_fit)
    return knn.predict(Z_all)


# ─── Clustering ──────────────────────────────────────────────────────────────

def _row(name: str, task: str, metric_label: str, value: float, display: str, elapsed: float, desc: str, suit: str,
         extra: Dict[str, Any], baseline: bool = False) -> Dict[str, Any]:
    return {
        "model_name": name, "problem_type": task, "metric_name": metric_label, "metric_value": round(value, 4),
        "metric_std": None, "metric_display": display, "training_time_sec": round(elapsed, 3), "description": desc,
        "suitability": suit, "is_recommended": False, "is_baseline": baseline, "extra": extra,
    }


def _cluster_profiles(X: pd.DataFrame, labels: np.ndarray, top: int = 4) -> List[Dict[str, Any]]:
    """Size and the features that most set each cluster apart (difference from the overall mean, in std units)."""
    overall_mean, overall_std = X.mean(), X.std().replace(0, 1)
    binary = {c for c in X.columns if set(np.unique(X[c])) <= {0.0, 1.0}}
    out = []
    for cl in sorted(set(labels)):
        mask = labels == cl
        sub = X[mask]
        z = ((sub.mean() - overall_mean) / overall_std).sort_values(key=lambda s: -s.abs())
        traits = []
        for feat in z.index[:top]:
            if abs(z[feat]) < 0.2:
                continue
            if feat in binary:
                traits.append({"feature": feat, "z": round(float(z[feat]), 2), "cluster": round(float(sub[feat].mean()) * 100, 1),
                               "overall": round(float(overall_mean[feat]) * 100, 1), "unit": "%"})
            else:
                traits.append({"feature": feat, "z": round(float(z[feat]), 2), "cluster": round(float(sub[feat].mean()), 4),
                               "overall": round(float(overall_mean[feat]), 4), "unit": ""})
        out.append({"cluster": int(cl), "size": int(mask.sum()), "share": round(float(mask.mean()) * 100, 1),
                    "noise": bool(cl == -1), "traits": traits})
    return out


def benchmark_clustering(df_cleaned: pd.DataFrame, lineage: Optional[Dict[str, str]] = None
                         ) -> Tuple[List[Dict[str, Any]], str, str, int, Dict[str, Any], pd.DataFrame]:
    """lineage: cleaned column -> original source column (from the data dictionary), for the shuffled reference."""
    X = feature_frame(df_cleaned)
    if len(X) < 20:
        raise BenchmarkError("Clustering needs at least 20 rows.")
    Z_all = _scaled(X)
    idx = _sample_index(len(X), MAX_FIT_ROWS)
    Z = Z_all[idx]
    n = len(Z)
    ks = _k_range(n)
    task, label = "clustering", "Silhouette"

    candidates: List[Tuple[str, str, str, Callable[[], Tuple[np.ndarray, Any, Dict[str, Any]]]]] = []

    def by_silhouette(make: Callable[[int], Any]) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
        best = (-2.0, None, None, None)
        for k in ks:
            model = make(k)
            labels = model.fit_predict(Z)
            s = _silhouette(Z, labels)
            if s is not None and s > best[0]:
                best = (s, labels, model, k)
        if best[1] is None:
            raise ValueError("no k produced two or more clusters")
        return best[1], best[2], {"k_selected_by": "silhouette", "k_tried": [ks[0], ks[-1]]}

    def gmm() -> Tuple[np.ndarray, Any, Dict[str, Any]]:
        cov = "full" if Z.shape[1] <= 20 else "diag"
        fits = []
        for k in ks:
            m = GaussianMixture(n_components=k, covariance_type=cov, random_state=42, n_init=1).fit(Z)
            fits.append((m.bic(Z), k, m))
        bic, k, m = min(fits, key=lambda t: t[0])
        return m.predict(Z), m, {"k_selected_by": "BIC", "bic": round(float(bic), 1)}

    def dbscan() -> Tuple[np.ndarray, Any, Dict[str, Any]]:
        min_samples = int(min(20, max(5, 2 * Z.shape[1])))
        kdist = np.sort(NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)[0][:, -1])
        best = (-2.0, None, None, None)
        for q in (0.5, 0.65, 0.8, 0.9, 0.95):
            eps = float(np.quantile(kdist, q))
            if eps <= 0:
                continue
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(Z)
            s = _silhouette(Z, labels)
            noise = float((labels == -1).mean())
            if s is not None and noise <= 0.5 and s > best[0]:
                best = (s, labels, model, eps)
        if best[1] is None:
            raise ValueError("no density threshold produced two or more clusters")
        return best[1], best[2], {"eps": round(best[3], 4), "min_samples": min_samples}

    def hdbscan() -> Tuple[np.ndarray, Any, Dict[str, Any]]:
        from sklearn.cluster import HDBSCAN
        model = HDBSCAN(min_cluster_size=max(5, n // 50))
        return model.fit_predict(Z), model, {"min_cluster_size": max(5, n // 50)}

    candidates = [
        ("K-Means", "Partitions rows into k compact, round groups around centroids. Fast and the usual first choice.", "High",
         lambda: by_silhouette(lambda k: KMeans(n_clusters=k, n_init=10, random_state=42))),
        ("Gaussian Mixture", "Soft, elliptical clusters with probabilities; k chosen by the Bayesian information criterion.", "High", gmm),
        ("Agglomerative (Ward)", "Merges rows bottom-up into a hierarchy, then cuts it at the best k. Good for nested structure.", "Moderate",
         lambda: by_silhouette(lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward"))),
        ("BIRCH", "Builds a compact clustering-feature tree first; scales to large data.", "Moderate",
         lambda: by_silhouette(lambda k: Birch(n_clusters=k))),
        ("DBSCAN", "Finds dense regions of any shape and labels sparse points as noise; no k needed.", "Moderate", dbscan),
        ("HDBSCAN", "Hierarchical density clustering: variable-density groups plus noise, no k or radius needed.", "High", hdbscan),
    ]

    results, fitted, errors = [], {}, []
    for name, desc, suit, run in candidates:
        start = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels, model, info = run()
            s = _silhouette(Z, labels)
            if s is None:
                raise ValueError("fewer than two clusters of a meaningful size")
            labels = _without_tiny(labels)
            mask = labels >= 0
            noise = float((~mask).mean())
            n_clusters = len(set(labels[mask]))
            extra = {
                "clusters": n_clusters,
                "noise_pct": round(noise * 100, 1),
                "davies_bouldin": round(float(davies_bouldin_score(Z[mask], labels[mask])), 3),
                "calinski_harabasz": round(float(calinski_harabasz_score(Z[mask], labels[mask])), 1),
                "sizes": sorted([int((labels == c).sum()) for c in set(labels[mask])], reverse=True),
                **info,
            }
            row = _row(name, task, label, s, f"{s:.3f}", time.time() - start, desc, suit, extra)
            row["_labels"] = labels
            results.append(row)
            fitted[name] = model
        except Exception as e:
            logger.warning(f"Clustering model '{name}' failed: {e}")
            errors.append(f"{name}: {e}")
    if not results:
        raise BenchmarkError(f"No clustering method could find two or more groups. First error: {errors[0] if errors else 'unknown'}")

    results.sort(key=lambda r: r["metric_value"], reverse=True)
    for i, r in enumerate(results):
        r["rank"], r["is_recommended"] = i + 1, i == 0
    best = results[0]

    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        null, null_k = _best_kmeans_silhouette(
            _shuffle_columns(Z, column_groups(X, sources=lineage, originals=[str(c) for c in df_cleaned.columns])), ks)
    results.append({**_row("Reference: same data, columns shuffled", task, label, null, f"{null:.3f}", time.time() - start,
                           "K-Means on the data with every column shuffled independently: same distributions, no real groups. "
                           "Real clusters must beat it clearly.", "Baseline", {"clusters": null_k}, baseline=True),
                    "rank": len(results) + 1})

    gap = best["metric_value"] - null
    if gap >= USEFUL_GAP and best["metric_value"] >= USEFUL_SILHOUETTE:
        verdict = "useful"
    elif gap >= WEAK_GAP:
        verdict = "weak"
    else:
        verdict = "no_signal"
    if verdict == "no_signal":
        for r in results:
            r["is_recommended"] = False

    labels_fit = best.pop("_labels")
    for r in results:
        r.pop("_labels", None)
    labels_all = _without_tiny(_assign_all(Z_all, Z, labels_fit, fitted[best["model_name"]]))
    profiles = _cluster_profiles(X, labels_all)
    k = best["extra"]["clusters"]

    if verdict == "no_signal":
        summary = (f"No real group structure: the best method, {best['model_name']}, reaches silhouette {best['metric_display']}, "
                   f"no better than the same data with its columns shuffled ({null:.3f}). Any clusters here would be arbitrary "
                   f"cuts of one continuous cloud; do not use them for segmentation.")
    else:
        summary = (f"Top Recommended Model: **{best['model_name']}** found {k} cluster(s) with silhouette **{best['metric_display']}** "
                   f"(Davies-Bouldin {best['extra']['davies_bouldin']}, lower is better) on {n} rows and {Z.shape[1]} features. "
                   f"The shuffled-columns reference scores {null:.3f}"
                   + (", so the groups are real and well separated." if verdict == "useful"
                      else ", so the groups exist but overlap; treat them as soft segments."))
    if errors:
        summary += f" {len(errors)} method(s) failed and were excluded."
    evaluation = {
        "task": task, "metric": "silhouette", "metric_label": label, "higher_is_better": True, "cv": "none", "n_splits": 0,
        "time_column": None, "rows_used": int(n), "rows_available": int(len(X)), "skipped_models": [], "source": "default",
        "reason": "Silhouette (-1..1) measures how much closer rows are to their own cluster than to the next one. Rows left "
                  "as noise, or in clusters under 3% of the data (a few outliers split off), do not count as a segment and "
                  "discount the score.",
        "verdict": verdict, "lift_over_baseline": round(gap, 4), "clusters": profiles, "k": int(k),
        "features": list(X.columns)[:200],
    }
    assignments = df_cleaned.copy()
    assignments["cluster"] = labels_all
    return results, label, summary, 0, evaluation, assignments


# ─── Anomaly detection ───────────────────────────────────────────────────────

def _robust_z(X: pd.DataFrame) -> pd.DataFrame:
    med = X.median()
    mad = (X - med).abs().median().replace(0, np.nan)
    z = 0.6745 * (X - med) / mad
    return z.fillna(0.0)


def _contamination(X: pd.DataFrame) -> float:
    continuous = [c for c in X.columns if X[c].nunique() > 10]
    if not continuous:
        return 0.05
    share = float((_robust_z(X[continuous]).abs() > 3.5).any(axis=1).mean())
    return float(min(0.1, max(0.01, share)))


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def benchmark_anomaly(df_cleaned: pd.DataFrame) -> Tuple[List[Dict[str, Any]], str, str, int, Dict[str, Any], pd.DataFrame]:
    X = feature_frame(df_cleaned)
    if len(X) < 30:
        raise BenchmarkError("Anomaly detection needs at least 30 rows.")
    Z_all = _scaled(X)
    idx = _sample_index(len(X), MAX_FIT_ROWS)
    Z = Z_all[idx]
    c = _contamination(X)
    n_flag = max(1, int(round(c * len(X))))
    task, label = "anomaly_detection", "Agreement (Jaccard)"
    rz = _robust_z(X)

    def top_share(scores: np.ndarray) -> np.ndarray:
        flags = np.zeros(len(scores), dtype=bool)
        flags[np.argsort(-scores)[:n_flag]] = True
        return flags

    detectors: List[Tuple[str, str, str, Callable[[], np.ndarray]]] = [
        ("Isolation Forest", "Random trees isolate unusual rows in few splits. Robust, fast, works on any feature mix.", "High",
         lambda: -IsolationForest(contamination=c, random_state=42, n_estimators=200).fit(Z).score_samples(Z_all)),
        ("Local Outlier Factor", "Flags rows that are much less dense than their neighbours: catches local, contextual anomalies.", "High",
         lambda: -LocalOutlierFactor(n_neighbors=min(20, len(Z) - 1), novelty=True, contamination=c).fit(Z).score_samples(Z_all)),
        ("One-Class SVM", "Learns a boundary around normal rows in a kernel space; anything outside is anomalous.", "Moderate",
         lambda: -OneClassSVM(nu=c, gamma="scale").fit(Z).score_samples(Z_all)),
        ("Robust z-score (MAD)", "Largest per-column deviation from the median in robust standard units. Simple and transparent.", "Moderate",
         lambda: rz.abs().max(axis=1).to_numpy()),
    ]
    if Z.shape[1] <= 30 and len(Z) > 5 * Z.shape[1]:
        detectors.append(("Elliptic Envelope", "Fits a robust Gaussian ellipse to the data; rows far outside it are anomalies.", "Moderate",
                          lambda: -EllipticEnvelope(contamination=c, random_state=42).fit(Z).score_samples(Z_all)))

    scores, flags, timings, errors = {}, {}, {}, []
    for name, _, _, run in detectors:
        start = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = np.asarray(run(), dtype=float)
            if not np.isfinite(s).all():
                s = np.nan_to_num(s, nan=0.0, posinf=np.nanmax(s[np.isfinite(s)]), neginf=0.0)
            scores[name], flags[name], timings[name] = s, top_share(s), time.time() - start
        except Exception as e:
            logger.warning(f"Anomaly detector '{name}' failed: {e}")
            errors.append(f"{name}: {e}")
    if len(flags) < 2:
        raise BenchmarkError(f"Fewer than two anomaly detectors could run. First error: {errors[0] if errors else 'unknown'}")

    names = list(flags)
    votes = np.sum([flags[nm] for nm in names], axis=0)
    consensus = votes >= max(2, int(np.ceil(len(names) / 2)))
    results = []
    for name, desc, suit, _ in detectors:
        if name not in flags:
            continue
        others = [flags[o] for o in names if o != name]
        loo = np.sum(others, axis=0) >= max(1, int(np.ceil(len(others) / 2)))
        agreement = _jaccard(flags[name], loo)
        results.append(_row(name, task, label, agreement, f"{agreement:.2f}", timings[name], desc, suit,
                            {"flagged": int(flags[name].sum()), "flagged_pct": round(float(flags[name].mean()) * 100, 2),
                             "overlap_with_consensus": int(np.logical_and(flags[name], consensus).sum())}))
    results.sort(key=lambda r: r["metric_value"], reverse=True)
    for i, r in enumerate(results):
        r["rank"], r["is_recommended"] = i + 1, i == 0

    rng = np.random.default_rng(42)
    rand = []
    for _ in range(50):
        f = np.zeros(len(X), dtype=bool)
        f[rng.choice(len(X), n_flag, replace=False)] = True
        rand.append(_jaccard(f, consensus))
    base = float(np.mean(rand))
    results.append({**_row("Reference: flag the same number of rows at random", task, label, base, f"{base:.2f}", 0.0,
                           "What agreement looks like by pure chance. Detectors must agree far more than this.", "Baseline",
                           {"flagged": n_flag}, baseline=True), "rank": len(results) + 1})

    best = results[0]
    if best["metric_value"] >= USEFUL_AGREEMENT and consensus.any():
        verdict = "useful"
    elif best["metric_value"] >= WEAK_AGREEMENT and consensus.any():
        verdict = "weak"
    else:
        verdict = "no_signal"
    if verdict == "no_signal":
        for r in results:
            r["is_recommended"] = False

    ranks = np.mean([pd.Series(scores[nm]).rank(pct=True).to_numpy() for nm in names], axis=0)
    order = [i for i in np.argsort(-ranks) if consensus[i]][:12]
    anomalies = []
    for i in order:
        zrow = rz.iloc[i]
        reasons = zrow.abs().sort_values(ascending=False).index[:3]
        anomalies.append({
            "row": int(df_cleaned.index[i]) + 1 if isinstance(df_cleaned.index[i], (int, np.integer)) else str(df_cleaned.index[i]),
            "score": round(float(ranks[i]), 3), "votes": int(votes[i]), "of": len(names),
            "reasons": [{"feature": f, "value": round(float(X.iloc[i][f]), 4), "robust_z": round(float(zrow[f]), 2)}
                        for f in reasons if abs(zrow[f]) >= 1.0],
        })

    if verdict == "no_signal":
        summary = (f"No consistent anomalies: the detectors disagree about which rows are unusual (best agreement "
                   f"{best['metric_display']} vs {base:.2f} for random picks). The data has no clear outliers, or they depend "
                   f"entirely on the method; do not act on these flags without review.")
    else:
        summary = (f"Top Recommended Model: **{best['model_name']}** agrees best with the other detectors (Jaccard "
                   f"**{best['metric_display']}** vs {base:.2f} at random). {int(consensus.sum())} row(s) "
                   f"({round(float(consensus.mean()) * 100, 1)}%) are flagged by a majority of {len(names)} detectors; "
                   f"expected contamination was estimated at {round(c * 100, 1)}% from robust z-scores.")
    if errors:
        summary += f" {len(errors)} detector(s) failed and were excluded."
    evaluation = {
        "task": task, "metric": "agreement", "metric_label": label, "higher_is_better": True, "cv": "none", "n_splits": 0,
        "time_column": None, "rows_used": int(len(X)), "rows_available": int(len(X)), "skipped_models": [], "source": "default",
        "reason": "Without labels, a detector is trusted when it flags the same rows as the other detectors (leave-one-out "
                  "majority, Jaccard overlap). Each flags its top contamination share of rows.",
        "verdict": verdict, "lift_over_baseline": round(best["metric_value"] - base, 4), "contamination": round(c, 4),
        "consensus_count": int(consensus.sum()), "anomalies": anomalies, "detectors": names,
    }
    assignments = df_cleaned.copy()
    assignments["is_anomaly"] = consensus.astype(int)
    assignments["anomaly_score"] = np.round(ranks, 4)
    assignments["anomaly_votes"] = votes.astype(int)
    return results, label, summary, 0, evaluation, assignments


def run_unsupervised(df_cleaned: pd.DataFrame, problem_type: str, lineage: Optional[Dict[str, str]] = None):
    if problem_type == "clustering":
        return benchmark_clustering(df_cleaned, lineage)
    if problem_type == "anomaly_detection":
        return benchmark_anomaly(df_cleaned)
    raise BenchmarkError(f"Unknown unsupervised task '{problem_type}'.")
