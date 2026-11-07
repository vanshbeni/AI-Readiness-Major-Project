"""Export an augmented training split. The holdout is original rows only."""
from __future__ import annotations

import csv
import io
import json
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from app.engine.augment_image import variants_for_image
from app.engine.augment_text import AugmentError, variant_budgets, variants_for_text

MIN_CLASS_FOR_HOLDOUT = 5


def _zip_write(archive: zipfile.ZipFile, name: str, data: bytes | str) -> None:
    info = zipfile.ZipInfo(filename=name, date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    payload = data.encode("utf-8") if isinstance(data, str) else data
    archive.writestr(info, payload)


def _csv(rows: list[list]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue()


def _versions() -> dict:
    import numpy
    import pandas
    import sklearn
    from PIL import Image as PilImage

    return {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "pillow": PilImage.__version__,
    }


def holdout_split(df: pd.DataFrame, target: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """80/20 stratified split. Classes with fewer than 5 rows stay in train."""
    labels = df[target].astype(str)
    counts = labels.value_counts()
    small = sorted(label for label, count in counts.items() if int(count) < MIN_CLASS_FOR_HOLDOUT)
    warnings = [
        f"Class '{label}' has fewer than {MIN_CLASS_FOR_HOLDOUT} rows, so it stayed entirely in the training split."
        for label in small
    ]
    tiny = df[labels.isin(small)]
    rest = df[~labels.isin(small)]
    if rest.empty:
        empty = df.iloc[0:0].copy()
        warnings.append("No class was large enough for a holdout, so every row stayed in train.")
        return df.copy(), empty, warnings
    stratify = rest[target].astype(str) if rest[target].astype(str).nunique() >= 2 else None
    train, test = train_test_split(
        rest,
        test_size=0.2,
        random_state=int(seed),
        stratify=stratify,
    )
    train = pd.concat([train, tiny], ignore_index=False)
    return train, test, warnings


def _with_ids(df: pd.DataFrame) -> pd.DataFrame:
    work = df.reset_index(drop=True).copy()
    work["_source_row_id"] = np.arange(len(work), dtype=int)
    return work


def export_text(df: pd.DataFrame, target: str, config: dict) -> tuple[str, str, dict]:
    config = {**config, "_target": target}
    column = config.get("text_column")
    if not column or column not in df.columns:
        raise AugmentError(f"Text column '{column}' was not found.")
    if target not in df.columns:
        raise AugmentError(f"Target column '{target}' was not found.")
    work = _with_ids(df.dropna(subset=[target]))
    train, test, warnings = holdout_split(work, target, int(config["seed"]))
    train = train.sort_values("_source_row_id")
    test = test.sort_values("_source_row_id")
    ops = [(item["name"], float(item["alpha"])) for item in config.get("ops") or []]
    budgets = variant_budgets(train[target].astype(str), config["strategy"], int(config["n_variants"]))
    synthetic = []
    for budget, (_, row) in zip(budgets, train.iterrows()):
        source = int(row["_source_row_id"])
        for text, names in variants_for_text(row[column], ops, budget, int(config["seed"]), source):
            payload = row.drop(labels=["_source_row_id"]).to_dict()
            payload[column] = text
            payload["is_augmented"] = 1
            payload["source_row_id"] = source
            payload["aug_ops"] = names
            synthetic.append(payload)
    original_columns = [name for name in df.columns if name in train.columns]
    originals = train[original_columns].copy()
    originals["is_augmented"] = 0
    originals["source_row_id"] = train["_source_row_id"].astype(int).tolist()
    originals["aug_ops"] = ""
    augmented = pd.concat([originals, pd.DataFrame(synthetic)], ignore_index=True) if synthetic else originals
    ordered = original_columns + ["is_augmented", "source_row_id", "aug_ops"]
    train_csv = augmented[ordered].to_csv(index=False, lineterminator="\n")
    holdout_csv = test[original_columns].to_csv(index=False, lineterminator="\n")
    manifest = _manifest(config, work, train, test, synthetic, warnings)
    return train_csv, holdout_csv, manifest


def export_images(df: pd.DataFrame, target: str, config: dict) -> tuple[bytes, dict]:
    config = {**config, "_target": target}
    if "image_path" not in df.columns or target not in df.columns:
        raise AugmentError("Image exports need image_path and the target column.")
    work = _with_ids(df.dropna(subset=[target]))
    train, test, warnings = holdout_split(work, target, int(config["seed"]))
    train = train.sort_values("_source_row_id")
    test = test.sort_values("_source_row_id")
    budgets = variant_budgets(train[target].astype(str), config["strategy"], int(config["n_variants"]))
    synthetic_rows = []
    buffer = io.BytesIO()
    train_rows = [["filename", "label", "is_augmented", "source_row_id", "aug_ops"]]
    test_rows = [["filename", "label"]]
    with zipfile.ZipFile(buffer, "w") as archive:
        for budget, (_, row) in zip(budgets, train.iterrows()):
            source = int(row["_source_row_id"])
            name = f"train/{source:04d}.png"
            with open(str(row["image_path"]), "rb") as handle:
                original = handle.read()
            _zip_write(archive, name, original)
            train_rows.append([name, str(row[target]), 0, source, ""])
            with Image.open(io.BytesIO(original)) as opened:
                image = opened.convert("RGB")
                image.load()
            for variant_index, (payload, ops) in enumerate(
                variants_for_image(image, config, budget, int(config["seed"]), source)
            ):
                aug_name = f"train/{source:04d}_aug{variant_index}.png"
                _zip_write(archive, aug_name, payload)
                train_rows.append([aug_name, str(row[target]), 1, source, ops])
                synthetic_rows.append(source)
        for _, row in test.iterrows():
            source = int(row["_source_row_id"])
            name = f"test/{source:04d}.png"
            with open(str(row["image_path"]), "rb") as handle:
                _zip_write(archive, name, handle.read())
            test_rows.append([name, str(row[target])])
        _zip_write(archive, "train_labels.csv", _csv(train_rows))
        _zip_write(archive, "test_labels.csv", _csv(test_rows))
    manifest = _manifest(config, work, train, test, synthetic_rows, warnings)
    return buffer.getvalue(), manifest


def _manifest(config, full, train, test, synthetic, warnings) -> dict:
    label_name = config.get("_target")
    return {
        "config": {key: value for key, value in config.items() if key != "_target"},
        "seed": int(config["seed"]),
        "counts_before": _counts(full, label_name),
        "counts_after_train": _counts_after(train, synthetic, label_name),
        "library_versions": _versions(),
        "split": {
            "train_original": int(len(train)),
            "train_synthetic": int(len(synthetic)),
            "holdout": int(len(test)),
        },
        "warnings": warnings,
    }


def _counts(frame: pd.DataFrame, target: str | None) -> dict:
    if not target or target not in frame.columns or frame.empty:
        return {}
    counts = frame[target].astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _counts_after(train: pd.DataFrame, synthetic, target: str | None) -> dict:
    if not target or target not in train.columns:
        return {}
    counts = {str(key): int(value) for key, value in train[target].astype(str).value_counts().items()}
    if not synthetic:
        return dict(sorted(counts.items()))
    if synthetic and isinstance(synthetic[0], dict):
        for row in synthetic:
            label = str(row[target])
            counts[label] = counts.get(label, 0) + 1
    else:
        id_to_label = {int(row["_source_row_id"]): str(row[target]) for _, row in train.iterrows()}
        for source in synthetic:
            label = id_to_label[int(source)]
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def manifest_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
