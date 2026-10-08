import csv
import io
import math
import os
import re
import threading
import time
import logging
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
_CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


class DatasetParseError(ValueError):
    pass


def sanitize_filename(filename: str) -> str:
    """Strips any directory components and unsafe characters from a client-supplied filename."""
    name = os.path.basename((filename or "").replace("\\", "/"))
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return name[:150] or "dataset.csv"


def _decode_csv(content: bytes) -> str:
    for enc in _CSV_ENCODINGS:
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DatasetParseError("Could not decode the CSV file. Please save it as UTF-8.")


def _sniff_delimiter(text_sample: str) -> str:
    try:
        return csv.Sniffer().sniff(text_sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Forces string column names, fills blank headers and de-duplicates repeated names."""
    warnings: List[str] = []
    new_cols: List[str] = []
    seen: Dict[str, int] = {}
    for idx, col in enumerate(df.columns):
        name = str(col).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{idx + 1}"
        if name in seen:
            seen[name] += 1
            renamed = f"{name}_{seen[name]}"
            warnings.append(f"Duplicate column '{name}' renamed to '{renamed}'.")
            name = renamed
        else:
            seen[name] = 0
        new_cols.append(name)
    df = df.copy()
    df.columns = new_cols
    return df, warnings


def parse_uploaded_file(content: bytes, ext: str) -> Tuple[pd.DataFrame, List[str]]:
    """Parses raw upload bytes into a DataFrame. Returns (df, warnings)."""
    warnings: List[str] = []
    if not content:
        raise DatasetParseError("The uploaded file is empty.")

    try:
        if ext == ".csv":
            text_data = _decode_csv(content)
            delimiter = _sniff_delimiter(text_data[:20000])
            df = pd.read_csv(io.StringIO(text_data), sep=delimiter)
        else:
            engine = "openpyxl" if ext == ".xlsx" else "xlrd"
            sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, engine=engine)
            non_empty = {name: frame for name, frame in sheets.items() if not frame.dropna(how="all").empty}
            if not non_empty:
                raise DatasetParseError("The Excel workbook has no data in any sheet.")
            sheet_name, df = next(iter(non_empty.items()))
            if len(sheets) > 1:
                warnings.append(f"Workbook has {len(sheets)} sheets; using the first non-empty sheet '{sheet_name}'.")
    except DatasetParseError:
        raise
    except ImportError as e:
        raise DatasetParseError(f"Missing parser dependency: {e}")
    except Exception as e:
        raise DatasetParseError(f"Failed to parse uploaded file: {e}")

    df = df.dropna(how="all").dropna(axis=1, how="all")
    df, col_warnings = normalize_columns(df)
    warnings.extend(col_warnings)
    df = df.replace([np.inf, -np.inf], np.nan)
    validate_dataframe_shape(df)
    return df, warnings


def validate_dataframe_shape(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    if cols < 2:
        raise DatasetParseError(
            f"Dataset has only {cols} usable column(s). At least 2 are needed (one target and one feature). "
            "If this is a CSV, check that it uses a standard delimiter."
        )
    if rows < settings.MIN_ROWS:
        raise DatasetParseError(f"Dataset has only {rows} data row(s). At least {settings.MIN_ROWS} are required.")


def read_dataset(path: str, nrows: int = None) -> pd.DataFrame:
    """Reads a stored dataset. Raw uploads are stored as normalized UTF-8 CSV at ingestion time."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, nrows=nrows)
    else:
        df = pd.read_excel(path, nrows=nrows)
    df.columns = [str(c) for c in df.columns]
    return df


def json_safe(value: Any) -> Any:
    """Recursively converts NaN/inf to None and numpy/pandas scalars to plain Python types."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    return value


def remove_files(*paths: str) -> None:
    for p in paths:
        if p and os.path.isfile(p):
            try:
                os.remove(p)
            except OSError as e:
                logger.warning(f"Could not remove file {p}: {e}")


def cleanup_expired_files() -> int:
    """Deletes stored files older than FILE_RETENTION_DAYS. Returns the number of files removed."""
    if settings.FILE_RETENTION_DAYS <= 0:
        return 0
    cutoff = time.time() - settings.FILE_RETENTION_DAYS * 86400
    removed = 0
    for folder in (settings.RAW_DATA_DIR, settings.CLEANED_DATA_DIR, settings.ARTIFACTS_DIR):
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if name == ".gitkeep" or not os.path.isfile(path):
                continue
            if os.path.getmtime(path) < cutoff:
                remove_files(path)
                removed += 1
    return removed


_locks_guard = threading.Lock()
_dataset_locks: Dict[str, threading.Lock] = {}


@contextmanager
def dataset_lock(dataset_id: str):
    """Rejects concurrent heavy operations on the same dataset instead of letting them race."""
    with _locks_guard:
        lock = _dataset_locks.setdefault(dataset_id, threading.Lock())
    if not lock.acquire(blocking=False):
        from fastapi import HTTPException
        raise HTTPException(status_code=409, detail="Another operation is already running for this dataset. Please wait.")
    try:
        yield
    finally:
        lock.release()
