"""Leak-free augmentation benchmark. Augmented rows are built only from each training fold."""
from __future__ import annotations

import io
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from app.engine.augment_image import budgets_for, image_features, variants_for_image
from app.engine.augment_text import AugmentError, variant_budgets, variants_for_text

TEXT_CAP = 2000
IMAGE_CAP = 1000
NOTE = "Augmentation is applied only to training folds."


def _labels(series: pd.Series) -> pd.Series:
    return series.astype(str)


def sample_rows(df: pd.DataFrame, target: str, cap: int, seed: int) -> pd.DataFrame:
    """Deterministic stratified cap. Original positions are kept on _source_row_id."""
    work = df.reset_index(drop=True).copy()
    work["_source_row_id"] = np.arange(len(work), dtype=int)
    if len(work) <= cap:
        return work
    labels = _labels(work[target])
    counts = labels.value_counts()
    n_classes = len(counts)
    if int(cap) < n_classes:
        raise AugmentError(f"Cannot sample {cap} rows across {n_classes} classes.")
    floor = 2 if cap >= 2 * n_classes else 1
    alloc = {label: min(int(count), floor) for label, count in counts.items()}
    capacity = {label: int(counts[label]) - alloc[label] for label in counts.index}
    remaining = cap - sum(alloc.values())
    total = float(counts.sum())
    shares = {label: remaining * (float(counts[label]) / total) for label in counts.index}
    for label in counts.index:
        give = min(capacity[label], int(np.floor(shares[label])))
        alloc[label] += give
        capacity[label] -= give
    leftover = cap - sum(alloc.values())
    order = sorted(counts.index, key=lambda label: (-(shares[label] - np.floor(shares[label])), str(label)))
    while leftover > 0:
        progressed = False
        for label in order:
            if leftover <= 0:
                break
            if capacity[label] <= 0:
                continue
            alloc[label] += 1
            capacity[label] -= 1
            leftover -= 1
            progressed = True
        if not progressed:
            break
    pieces = []
    for label in sorted(alloc, key=str):
        block = work.loc[labels == label]
        order_ids = sorted(block["_source_row_id"].tolist(), key=lambda row_id: ((int(row_id) * 17 + int(seed)) % 1000003, int(row_id)))
        chosen = set(order_ids[: alloc[label]])
        pieces.append(block[block["_source_row_id"].isin(chosen)])
    return pd.concat(pieces, ignore_index=True).sort_values("_source_row_id").reset_index(drop=True)


def _verdict(delta: float, spread: float) -> str:
    if abs(delta) <= spread + 1e-12:
        return "no meaningful change"
    return "augmentation helped" if delta > 0 else "augmentation hurt"


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def _score(model, train_x, train_y, valid_x, valid_y) -> float:
    fitted = clone(model)
    fitted.fit(train_x, train_y)
    predicted = fitted.predict(valid_x)
    return float(f1_score(valid_y, predicted, average="macro", zero_division=0))


def _text_models(seed: int):
    return [
        ("Logistic Regression", LogisticRegression(max_iter=400, random_state=seed), False),
        ("Linear SVC", LinearSVC(random_state=seed, dual="auto", max_iter=2000), False),
        ("Complement Naive Bayes", ComplementNB(), False),
        ("Most-frequent dummy", DummyClassifier(strategy="most_frequent"), True),
    ]


def _image_models(seed: int):
    return [
        ("Logistic Regression", LogisticRegression(max_iter=400, random_state=seed), False),
        ("Random Forest", RandomForestClassifier(n_estimators=40, max_depth=8, random_state=seed, n_jobs=1), False),
        ("Most-frequent dummy", DummyClassifier(strategy="most_frequent"), True),
    ]


def _text_matrix(train_texts: list[str], valid_texts: list[str]):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(train_texts), vectorizer.transform(valid_texts)


def _synthetic_text(frame: pd.DataFrame, text_column: str, target: str, config: dict) -> tuple[list[str], list[str], list[int]]:
    ops = [(item["name"], float(item["alpha"])) for item in config.get("ops") or []]
    budgets = variant_budgets(_labels(frame[target]), config["strategy"], int(config["n_variants"]))
    texts: list[str] = []
    labels: list[str] = []
    sources: list[int] = []
    for budget, (_, row) in zip(budgets, frame.iterrows()):
        source = int(row["_source_row_id"])
        for variant, names in variants_for_text(row[text_column], ops, budget, int(config["seed"]), source):
            texts.append(variant)
            labels.append(str(row[target]))
            sources.append(source)
    return texts, labels, sources


def _synthetic_images(frame: pd.DataFrame, path_column: str, target: str, config: dict) -> tuple[list[bytes], list[str], list[int]]:
    budgets = budgets_for(_labels(frame[target]), config["strategy"], int(config["n_variants"]))
    images: list[bytes] = []
    labels: list[str] = []
    sources: list[int] = []
    for budget, (_, row) in zip(budgets, frame.iterrows()):
        source = int(row["_source_row_id"])
        with Image.open(str(row[path_column])) as opened:
            image = opened.convert("RGB")
            image.load()
        for payload, _names in variants_for_image(image, config, budget, int(config["seed"]), source):
            images.append(payload)
            labels.append(str(row[target]))
            sources.append(source)
    return images, labels, sources


def _prepare(df: pd.DataFrame, target: str, modality: str) -> tuple[pd.DataFrame, str, int, int]:
    if target not in df.columns:
        raise AugmentError(f"Target column '{target}' was not found.")
    feature = "text" if modality == "text" else "image_path"
    column = df.attrs.get("feature_column")
    if column is None or column not in df.columns:
        raise AugmentError("The feature column for augmentation was not found.")
    if column == target:
        raise AugmentError("The feature column and the target must be different.")
    labeled = df.dropna(subset=[target]).copy()
    if labeled.empty:
        raise AugmentError("The target column has no labels.")
    cap = TEXT_CAP if modality == "text" else IMAGE_CAP
    sampled = sample_rows(labeled, target, cap, int(df.attrs.get("seed") or 42))
    counts = _labels(sampled[target]).value_counts()
    if int(counts.min()) < 2 or len(counts) < 2:
        raise AugmentError("Every class needs at least 2 rows before a validation fold can be held out.")
    return sampled, column, len(sampled), len(labeled)


def benchmark_augmentation(df: pd.DataFrame, target: str, config: dict) -> dict:
    """Same StratifiedKFold for both arms. Validation rows are never augmented."""
    modality = config["modality"]
    seed = int(config["seed"])
    work = df.copy()
    work.attrs["feature_column"] = config.get("text_column") if modality == "text" else "image_path"
    work.attrs["seed"] = seed
    sampled, column, used, available = _prepare(work, target, modality)
    labels = _labels(sampled[target])
    n_splits = min(5, int(labels.value_counts().min()))
    if n_splits < 2:
        raise AugmentError("The smallest class has fewer than 2 rows, so a validation fold cannot be built.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model_specs = _text_models(seed) if modality == "text" else _image_models(seed)
    fold_scores = {name: {"baseline": [], "augmented": []} for name, _, _ in model_specs}
    audits = []
    synthetic_counts = []
    y = labels.to_numpy()
    for fold, (train_index, valid_index) in enumerate(splitter.split(np.zeros(len(sampled)), y)):
        train = sampled.iloc[train_index]
        valid = sampled.iloc[valid_index]
        valid_ids = [int(value) for value in valid["_source_row_id"].tolist()]
        if modality == "text":
            train_text = ["" if pd.isna(value) or not isinstance(value, str) else value for value in train[column].tolist()]
            valid_text = ["" if pd.isna(value) or not isinstance(value, str) else value for value in valid[column].tolist()]
            extra_text, extra_y, sources = _synthetic_text(train, column, target, config)
            base_x, base_v = _text_matrix(train_text, valid_text)
            aug_x, aug_v = _text_matrix(train_text + extra_text, valid_text)
            train_y = _labels(train[target]).tolist()
            aug_y = train_y + extra_y
            valid_y = _labels(valid[target]).tolist()
        else:
            train_png = [_read_png(path) for path in train[column].tolist()]
            valid_png = [_read_png(path) for path in valid[column].tolist()]
            extra_png, extra_y, sources = _synthetic_images(train, column, target, config)
            base_x = np.vstack([image_features(payload) for payload in train_png])
            base_v = np.vstack([image_features(payload) for payload in valid_png])
            aug_x = np.vstack([image_features(payload) for payload in train_png + extra_png])
            aug_v = base_v
            train_y = _labels(train[target]).tolist()
            aug_y = train_y + extra_y
            valid_y = _labels(valid[target]).tolist()
        leaked = sorted(set(sources) & set(valid_ids))
        if leaked:
            raise AugmentError("Augmented rows overlapped a validation fold.")
        synthetic_counts.append(len(sources))
        audits.append({
            "fold": fold,
            "validation_row_ids": valid_ids,
            "synthetic_source_ids": sources,
        })
        for name, model, _baseline in model_specs:
            fold_scores[name]["baseline"].append(_score(model, base_x, train_y, base_v, valid_y))
            fold_scores[name]["augmented"].append(_score(model, aug_x, aug_y, aug_v, valid_y))
    models = []
    real_deltas = []
    real_spreads = []
    for name, _model, is_baseline in model_specs:
        base = fold_scores[name]["baseline"]
        aug = fold_scores[name]["augmented"]
        deltas = [a - b for a, b in zip(aug, base)]
        delta = float(np.mean(deltas))
        spread = _std(deltas)
        models.append({
            "model": name,
            "baseline_mean": float(np.mean(base)),
            "baseline_std": _std(base),
            "augmented_mean": float(np.mean(aug)),
            "augmented_std": _std(aug),
            "delta": delta,
            "verdict": _verdict(delta, spread),
            "is_baseline": is_baseline,
        })
        if not is_baseline:
            real_deltas.append(delta)
            real_spreads.append(spread)
    overall_delta = float(np.mean(real_deltas)) if real_deltas else 0.0
    overall_spread = float(np.mean(real_spreads)) if real_spreads else 0.0
    verdict = _verdict(overall_delta, overall_spread)
    warnings = []
    if used < available:
        warnings.append(f"Scored on {used} of {available} rows (stratified cap).")
    return {
        "modality": modality,
        "metric": "F1-Score (Macro)",
        "seed": seed,
        "rows_used": used,
        "rows_available": available,
        "folds": n_splits,
        "synthetic_rows": int(round(float(np.mean(synthetic_counts)))) if synthetic_counts else 0,
        "synthetic_rows_per_fold": synthetic_counts,
        "models": models,
        "verdict": verdict,
        "verdict_detail": (
            f"{verdict}: mean macro-F1 change {overall_delta:+.4f} "
            f"against a fold-to-fold spread of {overall_spread:.4f}. {NOTE}"
        ),
        "evaluation_note": NOTE,
        "leakage_audit": audits,
        "warnings": warnings,
    }


def _read_png(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def preview_text(df: pd.DataFrame, target: str, config: dict, n: int) -> list[dict]:
    column = config["text_column"]
    ops = [(item["name"], float(item["alpha"])) for item in config.get("ops") or []]
    changed = []
    unchanged = []
    for index, row in df.reset_index(drop=True).iterrows():
        label = "" if pd.isna(row[target]) else str(row[target])
        original = row[column] if isinstance(row[column], str) else None
        variants = variants_for_text(row[column], ops, 1, int(config["seed"]), int(index))
        if variants:
            text, names = variants[0]
            changed.append({
                "source_row_id": int(index),
                "label": label,
                "original": original,
                "augmented": text,
                "aug_ops": names,
            })
            if len(changed) >= n:
                break
        elif len(unchanged) < n:
            note = "No variant differed from the original." if original else "Skipped: empty, non-text, or too short to change."
            unchanged.append({
                "source_row_id": int(index),
                "label": label,
                "original": original,
                "augmented": None,
                "aug_ops": "",
                "note": note,
            })
    return (changed + unchanged)[:n]


def preview_images(df: pd.DataFrame, target: str, config: dict, n: int) -> list[dict]:
    from app.engine.augment_image import thumbnail_png

    samples = []
    for index, row in df.reset_index(drop=True).iterrows():
        if len(samples) >= n:
            break
        with Image.open(str(row["image_path"])) as opened:
            image = opened.convert("RGB")
            image.load()
            original = thumbnail_png(_png_of(image))
        variants = variants_for_image(image, config, 1, int(config["seed"]), int(index))
        augmented = thumbnail_png(variants[0][0]) if variants else None
        samples.append({
            "source_row_id": int(index),
            "label": str(row[target]),
            "original_png": original,
            "augmented_png": augmented,
            "aug_ops": variants[0][1] if variants else "",
            "note": None if variants else "No variant differed from the original.",
        })
    return samples


def suggest(df: pd.DataFrame, target: str | None, problem_type: str | None, modality: str | None) -> dict:
    """Fixed templates. Statistics come from the file; wording does not."""
    labels = df[target].dropna().astype(str) if target and target in df.columns else pd.Series(dtype=str)
    counts = {str(key): int(value) for key, value in labels.value_counts().sort_index().items()}
    ratio = None
    if counts:
        ratio = round(max(counts.values()) / max(1, min(counts.values())), 3)
    text_columns = []
    if modality != "image":
        for name in df.columns:
            if name == target:
                continue
            series = df[name]
            if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
                continue
            texts = [value.strip() for value in series.tolist() if isinstance(value, str) and value.strip()]
            if len(texts) < 8:
                continue
            lengths = [len(value.split()) for value in texts]
            median_words = float(np.median(lengths))
            unique_pct = round(len(set(texts)) / len(texts), 3)
            if median_words < 4:
                continue
            text_columns.append({
                "name": str(name),
                "non_empty": len(texts),
                "median_words": median_words,
                "unique_pct": unique_pct,
            })
    suggestions = []
    if text_columns:
        column = text_columns[0]
        suggestions.append({
            "id": "text-column",
            "title": f"Augment the text in '{column['name']}'",
            "detail": (
                f"'{column['name']}' has {column['non_empty']} non-empty values, "
                f"a median of {column['median_words']:.0f} words, and {column['unique_pct']:.0%} distinct texts."
            ),
            "text_column": column["name"],
            "strategy": "balance" if ratio and ratio >= 3 else "multiply",
        })
    elif modality != "image":
        suggestions.append({
            "id": "no-text",
            "title": "No text column found",
            "detail": "Text augmentation needs a column whose values are sentences, not short category labels.",
        })
    if ratio is not None and ratio >= 3:
        suggestions.append({
            "id": "imbalance",
            "title": "Classes are uneven",
            "detail": f"The largest class is {ratio:.1f} times the smallest. A balance strategy oversamples the smaller classes.",
            "strategy": "balance",
        })
    smallest = min(counts.values()) if counts else 0
    if len(df) < 200 or (counts and smallest < 20):
        suggestions.append({
            "id": "small-dataset",
            "title": "This dataset is small",
            "detail": f"{len(df)} rows are available" + (f", and the smallest class has {smallest}." if counts else "."),
            "strategy": "multiply",
        })
    if modality == "image":
        suggestions.insert(0, {
            "id": "image-mode",
            "title": "Image augmentation is available",
            "detail": "Flips, small rotations, brightness, contrast, noise, and crops are applied only to training images.",
            "strategy": "balance" if ratio and ratio >= 3 else "multiply",
        })
    return {
        "modality": modality or "tabular",
        "target_column": target,
        "problem_type": problem_type,
        "row_count": int(len(df)),
        "class_counts": counts,
        "imbalance_ratio": ratio,
        "text_columns": text_columns,
        "suggestions": suggestions,
    }


def _png_of(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=9)
    return buffer.getvalue()
