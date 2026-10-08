import os
import re
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.dataset import (
    Dataset,
    DatasetSnapshot,
    HealthScore,
    ModelBenchmark,
    ProcessingJob,
)
from app.core.storage import remove_files

_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def get_client_id(x_client_id: str = Header(default=None, alias="X-Client-Id")) -> str:
    if not x_client_id or not _CLIENT_ID_RE.match(x_client_id):
        raise HTTPException(status_code=401, detail="Missing or invalid X-Client-Id header.")
    return x_client_id


def get_owned_dataset(
    dataset_id: str,
    client_id: str = Depends(get_client_id),
    db: Session = Depends(get_db),
) -> Dataset:
    db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    # Unknown and foreign datasets both return 404 so dataset IDs cannot be probed.
    if not db_dataset or (db_dataset.owner_token and db_dataset.owner_token != client_id):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if not db_dataset.raw_file_path or not os.path.exists(db_dataset.raw_file_path):
        raise HTTPException(status_code=410, detail="The dataset file has expired or was removed. Please upload it again.")
    return db_dataset


def get_snapshot(db: Session, dataset_id: str) -> DatasetSnapshot:
    db.flush()  # the session has autoflush disabled; make a snapshot added earlier visible to the query
    snap = db.query(DatasetSnapshot).filter(DatasetSnapshot.dataset_id == dataset_id).first()
    if not snap:
        snap = DatasetSnapshot(dataset_id=dataset_id)
        db.add(snap)
    return snap


def invalidate_execution(db: Session, db_dataset: Dataset) -> None:
    """Clears everything derived from a pipeline run (cleaned data, after-score, jobs, benchmarks)."""
    jobs = db.query(ProcessingJob).filter(ProcessingJob.dataset_id == db_dataset.id).all()
    for job in jobs:
        remove_files(job.script_file_path, job.report_file_path)
        db.delete(job)
    remove_files(db_dataset.cleaned_file_path)
    db_dataset.cleaned_file_path = None
    db.query(HealthScore).filter(HealthScore.dataset_id == db_dataset.id, HealthScore.stage == "after").delete()
    db.query(ModelBenchmark).filter(ModelBenchmark.dataset_id == db_dataset.id).delete()
    snap = get_snapshot(db, db_dataset.id)
    snap.execution = None
    snap.leaderboard = None
