"""
Dataset policy: approved per-column decisions (from the Understand agent or the user) and how they are applied.

Decisions never modify the stored raw file. They become deterministic prep steps (transforms) that run before
profiling, cleaning and the visual lab, and that are written into the exported pipeline script.
"""
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from app.core.storage import read_dataset
from app.engine import normalize as N
from app.engine import transforms as T
from app.models.dataset import Dataset, DatasetPolicy

PrepOp = Tuple[str, Dict[str, Any], str]


def get_policy(db: Session, dataset_id: str, create: bool = False) -> Optional[DatasetPolicy]:
    db.flush()
    pol = db.query(DatasetPolicy).filter(DatasetPolicy.dataset_id == dataset_id).first()
    if not pol and create:
        pol = DatasetPolicy(dataset_id=dataset_id, column_overrides={})
        db.add(pol)
        db.flush()
    return pol


def column_overrides(db: Session, dataset_id: str) -> Dict[str, Dict[str, Any]]:
    pol = get_policy(db, dataset_id)
    return dict(pol.column_overrides or {}) if pol else {}


def type_map(overrides: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    return {col: o["type"] for col, o in overrides.items() if o.get("type") and not o.get("ignore")}


def prep_ops(overrides: Dict[str, Dict[str, Any]]) -> List[PrepOp]:
    """Deterministic order per column: missing tokens -> numeric parsing -> day-first dates -> category merge."""
    ops: List[PrepOp] = []
    ignored: List[str] = []
    for col, o in overrides.items():
        if o.get("ignore"):
            ignored.append(col)
            continue
        if o.get("na_tokens"):
            ops.append(("replace_with_nan", {"column": col, "values": list(o["na_tokens"])},
                        f"Prep: treated {', '.join(repr(v) for v in o['na_tokens'][:5])} as missing in '{col}'."))
        if o.get("parse_numeric"):
            decimal = (o["parse_numeric"] or {}).get("decimal", ".") if isinstance(o["parse_numeric"], dict) else "."
            ops.append(("parse_numeric", {"column": col, "decimal": decimal},
                        f"Prep: parsed '{col}' as numbers (currency symbols and separators removed, decimal '{decimal}')."))
        if o.get("dayfirst"):
            ops.append(("parse_dayfirst", {"column": col}, f"Prep: read '{col}' as day-first dates (dd/mm/yyyy)."))
        if o.get("normalize"):
            mapping = o["normalize"].get("mapping", {}) if isinstance(o["normalize"], dict) else {}
            if mapping:
                ops.append(("map_values", {"column": col, "mapping": mapping},
                            f"Prep: merged {len(mapping)} spelling variant(s) in '{col}'."))
    if ignored:
        ops.append(("drop_columns", {"columns": ignored}, f"Prep: excluded column(s) {', '.join(ignored)} as decided."))
    return ops


PROTECTED_TYPES = {"categorical", "text", "id", "datetime", "boolean"}


def prepare(df: pd.DataFrame, overrides: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, List[PrepOp], List[Dict[str, Any]]]:
    """
    Approved column decisions, then automatic normalisation (placeholder text -> missing, numbers written as text ->
    numbers). Columns the user typed explicitly are left alone. Returns (df, prep ops for the script, auto steps).
    """
    ops = prep_ops(overrides)
    for func, kwargs, _ in ops:
        df = getattr(T, func)(df, **kwargs)
    protect = {c for c, o in overrides.items() if o.get("type") in PROTECTED_TYPES}
    auto = N.plan(df, protect)
    for st in auto:
        df = getattr(T, st["func"])(df, **st["kwargs"])
        ops.append((st["func"], st["kwargs"], st["description"]))
    return df, ops, auto


def apply_prep(df: pd.DataFrame, overrides: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    return prepare(df, overrides)[0]


def load_prepared(db: Session, ds: Dataset, path: Optional[str] = None) -> Tuple[pd.DataFrame, List[PrepOp], Dict[str, str]]:
    """Reads the raw (or given) file and applies the approved + automatic prep. Returns (df, prep_ops, type_overrides)."""
    overrides = column_overrides(db, ds.id)
    df = read_dataset(path or ds.raw_file_path)
    if path and path != ds.raw_file_path:
        return df, [], type_map(overrides)
    df, ops, _ = prepare(df, overrides)
    return df, ops, type_map(overrides)


category_mapping = N.category_mapping
