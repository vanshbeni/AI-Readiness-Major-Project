"""
Pure, parameterised data transformations.

This module is embedded verbatim into the exported pipeline script, so it must only depend on
re, warnings, numpy and pandas, and every function must be deterministic given its arguments.
"""
import re
import warnings

import numpy as np
import pandas as pd

UNIT_PATTERN = re.compile(r"^\s*\$?(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)?\s*$")
TOKEN_SPLIT_PATTERN = re.compile(r"[,;|]")


def safe_name(value) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower() or "value"


def parse_dates(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, format="mixed", errors="coerce")


def split_tokens(cell) -> list:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    return [t.strip() for t in TOKEN_SPLIT_PATTERN.split(str(cell)) if t.strip()]


def parse_unit(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.nan, "unknown"
    match = UNIT_PATTERN.match(str(cell))
    if not match:
        return np.nan, "unknown"
    return float(match.group(1)), (match.group(2) or "unit").lower()


def drop_rows_missing(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    return df[df[column].notna()].reset_index(drop=True)


def drop_duplicate_rows(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    subset = [c for c in subset if c in df.columns] or None
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.drop(columns=[c for c in columns if c in df.columns])


def extract_datetime(df: pd.DataFrame, column: str, fill_year: float, fill_month: float, fill_day: float) -> pd.DataFrame:
    if column not in df.columns:
        return df
    parsed = parse_dates(df[column])
    df = df.copy()
    df[f"{column}_year"] = parsed.dt.year.fillna(fill_year).astype(float)
    df[f"{column}_month"] = parsed.dt.month.fillna(fill_month).astype(float)
    df[f"{column}_day"] = parsed.dt.day.fillna(fill_day).astype(float)
    return df.drop(columns=[column])


def split_units(df: pd.DataFrame, column: str, fill_value: float) -> pd.DataFrame:
    if column not in df.columns:
        return df
    parsed = [parse_unit(v) for v in df[column]]
    df = df.copy()
    df[f"{column}_value"] = pd.Series([p[0] for p in parsed], index=df.index).fillna(fill_value).astype(float)
    df[f"{column}_unit"] = pd.Series([p[1] for p in parsed], index=df.index)
    return df.drop(columns=[column])


def multi_hot(df: pd.DataFrame, column: str, tokens: list) -> pd.DataFrame:
    if column not in df.columns:
        return df
    token_sets = df[column].apply(lambda cell: set(split_tokens(cell)))
    df = df.copy()
    for token in tokens:
        df[f"{column}_{safe_name(token)}"] = token_sets.apply(lambda s, t=token: 1.0 if t in s else 0.0)
    return df.drop(columns=[column])


def replace_negatives(df: pd.DataFrame, column: str, fill_value: float) -> pd.DataFrame:
    if column not in df.columns:
        return df
    df = df.copy()
    df[column] = df[column].mask(df[column] < 0, fill_value)
    return df


def clip_values(df: pd.DataFrame, column: str, lower: float, upper: float) -> pd.DataFrame:
    if column not in df.columns:
        return df
    df = df.copy()
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def fill_missing(df: pd.DataFrame, column: str, value, add_flag: bool = False) -> pd.DataFrame:
    if column not in df.columns:
        return df
    df = df.copy()
    if add_flag:
        df[f"has_{column}"] = df[column].notna().astype(float)
    df[column] = df[column].fillna(value)
    return df


def group_rare(df: pd.DataFrame, column: str, keep: list, other_label: str = "Other") -> pd.DataFrame:
    if column not in df.columns:
        return df
    df = df.copy()
    keep_set = set(str(k) for k in keep)
    df[column] = df[column].apply(lambda v: v if pd.isna(v) or str(v) in keep_set else other_label)
    return df


def one_hot(df: pd.DataFrame, column: str, categories: list) -> pd.DataFrame:
    """Creates indicator columns for the given categories (the first observed category is the baseline)."""
    if column not in df.columns:
        return df
    as_text = df[column].astype(str)
    df = df.copy()
    for cat in categories:
        df[f"{column}_{safe_name(cat)}"] = (as_text == str(cat)).astype(float)
    return df.drop(columns=[column])
