"""Augmentation routes. They never write the cleaned CSV or the Recommendation table."""
import base64
import io
import os
import zipfile

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.api.deps import get_owned_dataset
from app.core.database import get_db
from app.core.storage import dataset_lock, json_safe, read_dataset, sanitize_filename
from app.engine.augment_benchmark import (
    AugmentError,
    benchmark_augmentation,
    preview_images,
    preview_text,
    suggest,
)
from app.engine.augment_export import export_images, export_text, manifest_bytes
from app.models.dataset import AugmentationRun, Dataset, Objective
from app.schemas.dataset import (
    AugmentPreviewResponse,
    AugmentSuggestions,
    AugmentationResult,
    AugmentationRunRequest,
)

router = APIRouter()


def _latest(db: Session, dataset_id: str) -> AugmentationRun | None:
    return (
        db.query(AugmentationRun)
        .filter(AugmentationRun.dataset_id == dataset_id)
        .order_by(AugmentationRun.created_at.desc(), AugmentationRun.id.desc())
        .first()
    )


def _classification_objective(db: Session, dataset_id: str) -> Objective:
    obj = db.query(Objective).filter(Objective.dataset_id == dataset_id).first()
    if not obj or not (obj.target_column or "").strip():
        raise HTTPException(status_code=400, detail="Set a classification objective before augmenting.")
    if obj.problem_type != "classification":
        raise HTTPException(status_code=400, detail="Augmentation benchmarks classification only. Regression and unsupervised objectives are rejected.")
    return obj


def _check_modality(dataset: Dataset, requested: str) -> None:
    actual = dataset.modality or "tabular"
    if requested == "image" and actual != "image":
        raise HTTPException(status_code=400, detail="This dataset has no images. Upload a zip to augment images.")
    if requested == "text" and actual == "image":
        raise HTTPException(status_code=400, detail="This is an image dataset. Text augmentation does not apply.")


def _parse_ops(raw: str) -> list[dict]:
    if not raw.strip():
        return []
    ops = []
    for part in raw.split(","):
        name, _, alpha = part.partition(":")
        try:
            value = float(alpha) if alpha else 0.1
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Could not read alpha for '{name}'.") from exc
        ops.append({"name": name.strip(), "alpha": value})
    return ops


def _request_from_query(
    dataset: Dataset,
    db: Session,
    text_column: str | None,
    strategy: str | None,
    n_variants: int | None,
    ops: str | None,
    seed: int | None,
    horizontal_flip: bool | None,
    vertical_flip: bool | None,
    max_deg: float | None,
    brightness: float | None,
    contrast: float | None,
    noise_std: float | None,
    crop_scale: float | None,
) -> AugmentationRunRequest:
    latest = _latest(db, dataset.id)
    base = dict(latest.config) if latest and latest.config else {}
    modality = "image" if dataset.modality == "image" else "text"
    if ops is not None:
        base["ops"] = _parse_ops(ops)
    payload = {
        "modality": modality,
        "text_column": text_column if text_column is not None else base.get("text_column"),
        "strategy": strategy or base.get("strategy") or "balance",
        "n_variants": base.get("n_variants", 2) if n_variants is None else n_variants,
        "ops": base.get("ops") or [],
        "seed": base.get("seed", 42) if seed is None else seed,
        "horizontal_flip": base.get("horizontal_flip", True) if horizontal_flip is None else horizontal_flip,
        "vertical_flip": base.get("vertical_flip", False) if vertical_flip is None else vertical_flip,
        "max_deg": base.get("max_deg", 15) if max_deg is None else max_deg,
        "brightness": base.get("brightness", 0.15) if brightness is None else brightness,
        "contrast": base.get("contrast", 0.15) if contrast is None else contrast,
        "noise_std": base.get("noise_std", 6) if noise_std is None else noise_std,
        "crop_scale": base.get("crop_scale", 0.85) if crop_scale is None else crop_scale,
    }
    try:
        return AugmentationRunRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _b64(payload: bytes | None) -> str | None:
    if payload is None:
        return None
    return base64.b64encode(payload).decode("ascii")


@router.get("/{dataset_id}/augmentation/suggestions", response_model=AugmentSuggestions)
def augmentation_suggestions(
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    frame = read_dataset(db_dataset.raw_file_path)
    obj = db.query(Objective).filter(Objective.dataset_id == db_dataset.id).first()
    payload = suggest(
        frame,
        obj.target_column if obj else None,
        obj.problem_type if obj else None,
        db_dataset.modality,
    )
    latest = _latest(db, db_dataset.id)
    if latest is not None:
        payload["latest_config"] = latest.config
        payload["latest_result"] = latest.result
    return json_safe(payload)


@router.get("/{dataset_id}/augmentation/preview", response_model=AugmentPreviewResponse)
def augmentation_preview(
    n: int = Query(4, ge=1, le=8),
    text_column: str | None = None,
    strategy: str | None = None,
    n_variants: int | None = Query(None, ge=0, le=10),
    ops: str | None = None,
    seed: int | None = Query(None, ge=0, le=2_147_483_647),
    horizontal_flip: bool | None = None,
    vertical_flip: bool | None = None,
    max_deg: float | None = Query(None, ge=0, le=45),
    brightness: float | None = Query(None, ge=0, le=0.5),
    contrast: float | None = Query(None, ge=0, le=0.5),
    noise_std: float | None = Query(None, ge=0, le=30),
    crop_scale: float | None = Query(None, ge=0.5, le=1),
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    request = _request_from_query(
        db_dataset, db, text_column, strategy, n_variants, ops, seed,
        horizontal_flip, vertical_flip, max_deg, brightness, contrast, noise_std, crop_scale,
    )
    _check_modality(db_dataset, request.modality)
    obj = _classification_objective(db, db_dataset.id)
    frame = read_dataset(db_dataset.raw_file_path)
    config = request.model_dump()
    try:
        if request.modality == "text":
            samples = preview_text(frame, obj.target_column, config, n)
        else:
            samples = preview_images(frame, obj.target_column, config, n)
            for sample in samples:
                sample["original_png"] = _b64(sample.get("original_png"))
                sample["augmented_png"] = _b64(sample.get("augmented_png"))
    except AugmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return json_safe({"modality": request.modality, "samples": samples})


@router.post("/{dataset_id}/augmentation/run", response_model=AugmentationResult)
def run_augmentation(
    payload: AugmentationRunRequest,
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    _check_modality(db_dataset, payload.modality)
    obj = _classification_objective(db, db_dataset.id)
    with dataset_lock(db_dataset.id):
        frame = read_dataset(db_dataset.raw_file_path)
        try:
            result = benchmark_augmentation(frame, obj.target_column, payload.model_dump())
        except AugmentError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        stored = json_safe(result)
        db.add(AugmentationRun(
            dataset_id=db_dataset.id,
            config=json_safe(payload.model_dump()),
            result=stored,
        ))
        db.commit()
    return stored


def _export_bundle(db: Session, dataset: Dataset):
    run = _latest(db, dataset.id)
    if run is None or not run.config:
        raise HTTPException(status_code=400, detail="Run an augmentation benchmark before exporting.")
    obj = _classification_objective(db, dataset.id)
    _check_modality(dataset, run.config.get("modality"))
    frame = read_dataset(dataset.raw_file_path)
    try:
        if run.config.get("modality") == "image":
            blob, manifest = export_images(frame, obj.target_column, run.config)
            return blob, manifest, "zip"
        train_csv, holdout_csv, manifest = export_text(frame, obj.target_column, run.config)
    except AugmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        from app.engine.augment_export import _zip_write

        _zip_write(archive, "train_augmented.csv", train_csv)
        _zip_write(archive, "holdout_test.csv", holdout_csv)
    return buffer.getvalue(), manifest, "zip"


@router.get("/{dataset_id}/export/augmented-train")
def download_augmented_train(
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    with dataset_lock(db_dataset.id):
        blob, _manifest, _kind = _export_bundle(db, db_dataset)
    stem = os.path.splitext(sanitize_filename(db_dataset.filename))[0] or "dataset"
    return Response(
        content=blob,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{stem}_augmented_train.zip"'},
    )


@router.get("/{dataset_id}/export/augmented-manifest")
def download_augmented_manifest(
    db_dataset: Dataset = Depends(get_owned_dataset),
    db: Session = Depends(get_db),
):
    with dataset_lock(db_dataset.id):
        _blob, manifest, _kind = _export_bundle(db, db_dataset)
    return Response(content=manifest_bytes(json_safe(manifest)), media_type="application/json")
