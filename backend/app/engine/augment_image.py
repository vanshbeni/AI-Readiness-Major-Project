"""Pillow + NumPy image augmentation and safe zip ingestion. No pretrained models."""
from __future__ import annotations

import io
import os
import re
import zipfile
from typing import Sequence

import numpy as np
from PIL import Image, ImageEnhance, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 20_000_000

from app.engine.augment_text import MAX_VARIANTS, derive_rng, derive_seed_int, op_index, variant_budgets
from app.engine.augment_text import AugmentError

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MIN_IMAGES = 10
MIN_CLASSES = 2
MIN_PER_CLASS = 2
MAX_SIDE = 256
_GENERIC_DIRS = {"train", "test", "val", "valid", "validation", "images", "data", "dataset"}


def normalize_image(data: bytes) -> bytes:
    """RGB PNG whose longest side is at most 256 pixels."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as opened:
            image = opened.convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise AugmentError(f"corrupt image: {exc.__class__.__name__}") from exc
    width, height = image.size
    longest = max(width, height)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / float(longest)
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    return _png_bytes(image)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", compress_level=9)
    return buffer.getvalue()


def _safe_member(name: str) -> str:
    raw = (name or "").replace("\\", "/")
    if raw.startswith("/") or raw.startswith("~") or re.match(r"^[A-Za-z]:", raw):
        raise AugmentError(f"Rejected archive path '{name}'.")
    parts = []
    for part in raw.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise AugmentError(f"Rejected archive path '{name}'.")
        parts.append(part)
    if not parts:
        raise AugmentError(f"Rejected archive path '{name}'.")
    return "/".join(parts)


def _class_from_path(path: str) -> str | None:
    parent = os.path.dirname(path).replace("\\", "/")
    if not parent:
        return None
    bits = [bit for bit in parent.split("/") if bit and bit.lower() not in _GENERIC_DIRS and not bit.startswith(".")]
    if not bits:
        return None
    return bits[-1]


def read_image_archive(data: bytes, max_images: int) -> tuple[list[dict], list[str]]:
    """Validate a zip in memory. Returns normalized PNG records and warnings.

    Accepts labels.csv (filename, label) or a folder-per-class layout.
    Unsafe paths reject the whole archive. Corrupt images are skipped.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise AugmentError("The upload is not a valid zip file.") from exc
    warnings: list[str] = []
    members: dict[str, bytes] = {}
    labels_bytes: bytes | None = None
    with archive:
        for info in archive.infolist():
            if info.is_dir():
                _safe_member(info.filename.rstrip("/") or info.filename)
                continue
            path = _safe_member(info.filename)
            base = os.path.basename(path)
            if base.startswith(".") or path.split("/")[0] == "__MACOSX":
                continue
            if base.lower() == "labels.csv":
                labels_bytes = archive.read(info)
                continue
            ext = os.path.splitext(base)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                continue
            if len(members) >= max_images:
                raise AugmentError(f"The archive has more than {max_images} images.")
            members[path] = archive.read(info)

    labeled: list[tuple[str, str]] = []
    if labels_bytes is not None:
        labeled = _parse_labels(labels_bytes, set(members))
        warnings.append("Used labels.csv for class names.")
    else:
        for path in sorted(members):
            label = _class_from_path(path)
            if label:
                labeled.append((path, label))
        if not labeled:
            raise AugmentError("The zip needs a labels.csv (filename, label) or one folder per class.")

    records = []
    for index, (path, label) in enumerate(sorted(labeled, key=lambda item: (item[1], item[0]))):
        try:
            png = normalize_image(members[path])
        except AugmentError:
            warnings.append(f"Skipped corrupt image '{path}'.")
            continue
        records.append({"label": str(label), "source_name": path, "png": png, "filename": f"{index:04d}.png"})

    if len(records) < MIN_IMAGES:
        raise AugmentError(f"Need at least {MIN_IMAGES} readable images; found {len(records)}.")
    counts: dict[str, int] = {}
    for record in records:
        counts[record["label"]] = counts.get(record["label"], 0) + 1
    eligible = [label for label, count in counts.items() if count >= MIN_PER_CLASS]
    if len(eligible) < MIN_CLASSES:
        raise AugmentError(
            f"Need at least {MIN_CLASSES} classes with {MIN_PER_CLASS} or more images each."
        )
    return records, warnings


def _parse_labels(payload: bytes, names: set[str]) -> list[tuple[str, str]]:
    import csv

    text = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = payload.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise AugmentError("labels.csv could not be decoded.")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise AugmentError("labels.csv is missing a header row.")
    fields = {name.strip().lower(): name for name in reader.fieldnames}
    if "filename" not in fields or "label" not in fields:
        raise AugmentError("labels.csv must have columns filename and label.")
    found = []
    known = {name.replace("\\", "/"): name for name in names}
    basenames: dict[str, str] = {}
    for name in names:
        basenames.setdefault(os.path.basename(name), name)
    for row in reader:
        raw_name = (row.get(fields["filename"]) or "").strip().replace("\\", "/")
        label = (row.get(fields["label"]) or "").strip()
        if not raw_name or not label:
            continue
        _safe_member(raw_name)
        match = known.get(raw_name) or basenames.get(os.path.basename(raw_name))
        if match is None:
            continue
        found.append((match, label))
    if not found:
        raise AugmentError("labels.csv did not match any image in the archive.")
    return found


def _local_generator(seed: int, source_row_id: int, index: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(derive_seed_int(seed, source_row_id, index)))


def augment_image(image: Image.Image, config: dict, seed: int, source_row_id: int, variant: int, attempt: int = 0) -> tuple[Image.Image | None, str]:
    """One deterministic variant. Returns (None, '') when the pixels do not change."""
    current = image.convert("RGB")
    applied: list[str] = []
    slot = 0

    def rng():
        nonlocal slot
        drawn = derive_rng(seed, source_row_id, op_index(variant, attempt, slot))
        slot += 1
        return drawn

    if config.get("horizontal_flip") and rng().random() < 0.5:
        current = current.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        applied.append("horizontal_flip")
    if config.get("vertical_flip") and rng().random() < 0.5:
        current = current.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        applied.append("vertical_flip")
    max_deg = float(config.get("max_deg") or 0)
    if max_deg > 0:
        angle = rng().uniform(-max_deg, max_deg)
        if abs(angle) >= 0.5:
            current = current.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
            applied.append("rotation")
    brightness = float(config.get("brightness") or 0)
    if brightness > 0:
        factor = 1.0 + rng().uniform(-brightness, brightness)
        current = ImageEnhance.Brightness(current).enhance(factor)
        applied.append("brightness")
    contrast = float(config.get("contrast") or 0)
    if contrast > 0:
        factor = 1.0 + rng().uniform(-contrast, contrast)
        current = ImageEnhance.Contrast(current).enhance(factor)
        applied.append("contrast")
    noise_std = float(config.get("noise_std") or 0)
    if noise_std > 0:
        generator = _local_generator(seed, source_row_id, op_index(variant, attempt, slot))
        slot += 1
        arr = np.asarray(current, dtype=np.float64)
        arr = np.clip(arr + generator.normal(0.0, noise_std, size=arr.shape), 0, 255)
        current = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        applied.append("gaussian_noise")
    crop_scale = float(config.get("crop_scale") or 1)
    if 0 < crop_scale < 1:
        scale = rng().uniform(crop_scale, 1.0)
        width, height = current.size
        crop_w = max(1, int(round(width * scale)))
        crop_h = max(1, int(round(height * scale)))
        if crop_w < width or crop_h < height:
            left = rng().randrange(0, width - crop_w + 1)
            top = rng().randrange(0, height - crop_h + 1)
            current = current.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.Resampling.BICUBIC)
            applied.append("crop")
    if not applied:
        return None, ""
    before = np.asarray(image.convert("RGB"))
    after = np.asarray(current)
    if before.shape == after.shape and np.array_equal(before, after):
        return None, ""
    return current, ",".join(applied)


def variants_for_image(image: Image.Image, config: dict, n_variants: int, seed: int, source_row_id: int) -> list[tuple[bytes, str]]:
    n_variants = max(0, min(MAX_VARIANTS, int(n_variants)))
    produced: list[tuple[bytes, str]] = []
    seen = {_png_bytes(image)}
    attempt = 0
    while len(produced) < n_variants and attempt < n_variants * 6:
        variant, ops = augment_image(image, config, seed, source_row_id, len(produced), attempt)
        attempt += 1
        if variant is None:
            continue
        payload = _png_bytes(variant)
        if payload in seen:
            continue
        seen.add(payload)
        produced.append((payload, ops))
    return produced


def image_features(png: bytes) -> np.ndarray:
    """16x16 flattened RGB plus a per-channel 8-bin histogram."""
    with Image.open(io.BytesIO(png)) as opened:
        image = opened.convert("RGB").resize((16, 16), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float64) / 255.0
    hist = []
    for channel in range(3):
        counts, _ = np.histogram(array[:, :, channel], bins=8, range=(0.0, 1.0))
        hist.append(counts.astype(np.float64))
    histogram = np.concatenate(hist)
    total = histogram.sum()
    if total:
        histogram = histogram / total
    return np.concatenate([array.reshape(-1), histogram])


def thumbnail_png(png: bytes, size: int = 96) -> bytes:
    with Image.open(io.BytesIO(png)) as opened:
        image = opened.convert("RGB")
        image.thumbnail((size, size))
    return _png_bytes(image)


def budgets_for(labels: Sequence, strategy: str, n_variants: int) -> list[int]:
    return variant_budgets(labels, strategy, n_variants)
