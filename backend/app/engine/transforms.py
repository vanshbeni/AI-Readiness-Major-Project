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


def fresh_name(df: pd.DataFrame, name: str) -> str:
    """`name`, or `name_2`, `name_3`, ... if a column with that name already exists."""
    existing = {str(c) for c in df.columns}
    candidate, k = name, 2
    while candidate in existing:
        candidate = f"{name}_{k}"
        k += 1
    return candidate


def indicator_names(df: pd.DataFrame, column: str, values: list) -> list:
    """
    One output column name per value. Distinct values that sanitise to the same name
    ('Kepler' / 'kepler ', 'A+' / 'A-') get numeric suffixes instead of overwriting each other,
    and names already present in the frame are never reused.
    """
    used = {str(c) for c in df.columns if str(c) != column}
    names = []
    for v in values:
        base = f"{column}_{safe_name(v)}"
        name, k = base, 2
        while name in used:
            name = f"{base}_{k}"
            k += 1
        used.add(name)
        names.append(name)
    return names


def parse_dates(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # utc=True so mixed offsets ("...+05:30" next to "...Z") parse instead of failing the whole column.
        return pd.to_datetime(series, format="mixed", errors="coerce", utc=True)


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


def _token(value) -> str:
    """Canonical text for comparing cells with user-declared tokens ('-999', -999 and -999.0 all match)."""
    try:
        f = float(value)
        if np.isfinite(f):
            return str(int(f)) if f.is_integer() else repr(f)
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower()


def replace_with_nan(df: pd.DataFrame, column: str, values: list) -> pd.DataFrame:
    """Treats placeholder tokens ('?', 'unknown') and sentinel numbers (-999) as missing."""
    if column not in df.columns:
        return df
    tokens = {_token(v) for v in values}
    df = df.copy()
    col = df[column].astype(object)
    col = col.where(~col.map(lambda v: not pd.isna(v) and _token(v) in tokens), np.nan)
    numeric = pd.to_numeric(col, errors="coerce")
    df[column] = numeric if numeric.notna().sum() == col.notna().sum() else col
    return df


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALE_WORDS = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7}
MAGNITUDE_SUFFIX = re.compile(r"(?i)(?<=[\d\s.])(k|thousand|million|mn|billion|bn|lakhs?|crores?)\s*\)?\s*$")
MAGNITUDES = {"k": 1e3, "thousand": 1e3, "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9,
              "lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7}


def _number_word_tokens(text) -> list:
    return [t for t in re.findall(r"[a-z]+", str(text).lower()) if t != "and"]


def is_number_words(text) -> bool:
    """'thirty', 'SIXTY THOUSAND', 'one hundred and five' -> True."""
    tokens = _number_word_tokens(text)
    return (bool(tokens) and not any(ch.isdigit() for ch in str(text))
            and all(t in NUMBER_WORDS or t in SCALE_WORDS or t == "hundred" for t in tokens))


def words_to_number(text) -> float:
    if not is_number_words(text):
        return np.nan
    total, current = 0.0, 0.0
    for t in _number_word_tokens(text):
        if t in NUMBER_WORDS:
            current += NUMBER_WORDS[t]
        elif t == "hundred":
            current = max(current, 1) * 100
        else:
            total += max(current, 1) * SCALE_WORDS[t]
            current = 0.0
    return float(total + current)


def to_number(cell, decimal: str = "."):
    """'$12,345' -> 12345.0, '12,50' (decimal=',') -> 12.5, '(40)' -> -40.0, '60k' -> 60000.0,
    'SIXTY THOUSAND' -> 60000.0; unparseable -> NaN."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.nan
    if isinstance(cell, (int, float, np.number)) and not isinstance(cell, bool):
        return float(cell)
    s = str(cell).strip()
    if not any(ch.isdigit() for ch in s):
        return words_to_number(s)
    negative = s.startswith("(") and s.endswith(")")
    multiplier = 1.0
    magnitude = MAGNITUDE_SUFFIX.search(s)
    if magnitude:
        multiplier = MAGNITUDES[magnitude.group(1).lower()]
        s = s[:magnitude.start()]
    s = re.sub(r"[^\d.,\-]", "", s)
    s = s.replace(".", "").replace(",", ".") if decimal == "," else s.replace(",", "")
    try:
        value = float(s) * multiplier
    except ValueError:
        return np.nan
    return -abs(value) if negative else value


def strip_text(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Removes leading / trailing whitespace (' grace ' -> 'grace')."""
    if column not in df.columns:
        return df
    df = df.copy()
    df[column] = df[column].map(lambda v: v.strip() if isinstance(v, str) else v)
    return df


def parse_numeric(df: pd.DataFrame, column: str, decimal: str = ".") -> pd.DataFrame:
    if column not in df.columns:
        return df
    df = df.copy()
    df[column] = pd.Series([to_number(v, decimal) for v in df[column]], index=df.index, dtype=float)
    return df


def map_values(df: pd.DataFrame, column: str, mapping: dict) -> pd.DataFrame:
    """Replaces spelling variants with a canonical value (e.g. 'delhi ' / 'DELHI' -> 'Delhi')."""
    if column not in df.columns:
        return df
    df = df.copy()
    df[column] = df[column].map(lambda v: v if pd.isna(v) else mapping.get(str(v), mapping.get(str(v).strip(), v)))
    return df


def parse_dayfirst(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Rewrites day-first dates ('13/01/2024') as ISO strings so later date handling is unambiguous."""
    if column not in df.columns:
        return df
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(df[column], dayfirst=True, format="mixed", errors="coerce", utc=True)
    df = df.copy()
    has_time = bool(((parsed.dt.hour != 0) | (parsed.dt.minute != 0)).fillna(False).any())
    fmt = "%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d"
    df[column] = parsed.dt.strftime(fmt).where(parsed.notna(), df[column])
    return df


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
    df[fresh_name(df, f"{column}_year")] = parsed.dt.year.fillna(fill_year).astype(float)
    df[fresh_name(df, f"{column}_month")] = parsed.dt.month.fillna(fill_month).astype(float)
    df[fresh_name(df, f"{column}_day")] = parsed.dt.day.fillna(fill_day).astype(float)
    return df.drop(columns=[column])


def split_units(df: pd.DataFrame, column: str, fill_value: float) -> pd.DataFrame:
    if column not in df.columns:
        return df
    parsed = [parse_unit(v) for v in df[column]]
    df = df.copy()
    df[fresh_name(df, f"{column}_value")] = pd.Series([p[0] for p in parsed], index=df.index).fillna(fill_value).astype(float)
    df[fresh_name(df, f"{column}_unit")] = pd.Series([p[1] for p in parsed], index=df.index)
    return df.drop(columns=[column])


def multi_hot(df: pd.DataFrame, column: str, tokens: list) -> pd.DataFrame:
    if column not in df.columns:
        return df
    token_sets = df[column].apply(lambda cell: set(split_tokens(cell)))
    df = df.copy()
    for token, name in zip(tokens, indicator_names(df, column, tokens)):
        df[name] = token_sets.apply(lambda s, t=token: 1 if t in s else 0).astype(int)
    return df.drop(columns=[column])


CODE_PREFIX_PATTERN = re.compile(r"^\s*([A-Za-z]+)[\s\-_/.#]*\d")
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z']+")


def code_prefix(cell):
    """'C85' -> 'C', 'SKU-123' -> 'SKU', 'A/5 21171' -> 'A'; values without a letter prefix -> None."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    match = CODE_PREFIX_PATTERN.match(str(cell))
    return match.group(1).upper() if match else None


def word_set(cell) -> set:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return set()
    return {w.lower() for w in WORD_PATTERN.findall(str(cell))}


def extract_prefix(df: pd.DataFrame, column: str, keep: list) -> pd.DataFrame:
    """Adds '<column>_prefix': the letter prefix of a code if it is a kept prefix, 'Other' for the rest, 'Missing' for blanks."""
    if column not in df.columns:
        return df
    keep_set = set(keep)
    df = df.copy()
    prefixes = df[column].map(code_prefix)
    df[fresh_name(df, f"{column}_prefix")] = [
        "Missing" if pd.isna(v) else (p if p in keep_set else "Other") for v, p in zip(df[column], prefixes)
    ]
    return df


def word_flags(df: pd.DataFrame, column: str, words: list) -> pd.DataFrame:
    """One 0/1 column per frequent word (e.g. titles such as 'mr' / 'mrs' inside a name column)."""
    if column not in df.columns:
        return df
    sets = df[column].map(word_set)
    df = df.copy()
    for word, name in zip(words, indicator_names(df, f"{column}_word", words)):
        df[name] = sets.map(lambda s, w=word: 1 if w in s else 0).astype(int)
    return df


def infinite_to_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Infinite values (division by zero upstream, 'inf' text) become blanks, so they are imputed and flagged."""
    floats = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    if not floats:
        return df
    df = df.copy()
    df[floats] = df[floats].replace([np.inf, -np.inf], np.nan)
    return df


def negatives_to_missing(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impossible negative values become blanks, so they are imputed (and flagged) like any other gap."""
    if column not in df.columns:
        return df
    df = df.copy()
    df[column] = df[column].mask(df[column] < 0)
    return df


def replace_negatives(df: pd.DataFrame, column: str, fill_value: float) -> pd.DataFrame:
    if column not in df.columns:
        return df
    df = df.copy()
    df[column] = df[column].mask(df[column] < 0, fill_value)
    return df


def clip_values(df: pd.DataFrame, column: str, lower: float, upper: float, add_flag: bool = False) -> pd.DataFrame:
    """Caps values to [lower, upper]; add_flag records which rows were capped in '<column>_was_capped'."""
    if column not in df.columns:
        return df
    df = df.copy()
    if add_flag:
        df[fresh_name(df, f"{column}_was_capped")] = ((df[column] < lower) | (df[column] > upper)).astype(int)
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def presence_flag(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Replaces a mostly-empty column by 'has_<column>' (1 = a value was recorded)."""
    if column not in df.columns:
        return df
    df = df.copy()
    df[fresh_name(df, f"has_{column}")] = df[column].notna().astype(int)
    return df.drop(columns=[column])


def fill_missing(df: pd.DataFrame, column: str, value, add_flag: bool = False, flag_style: str = "has") -> pd.DataFrame:
    """Fills blanks with `value`. add_flag keeps a record of which rows were blank:
    flag_style 'has' -> 'has_<column>' (1 = value present), 'missing' -> '<column>_was_missing' (1 = was blank)."""
    if column not in df.columns:
        return df
    df = df.copy()
    if add_flag:
        if flag_style == "missing":
            df[fresh_name(df, f"{column}_was_missing")] = df[column].isna().astype(int)
        else:
            df[fresh_name(df, f"has_{column}")] = df[column].notna().astype(int)
    df[column] = df[column].fillna(value)
    return df


def fill_by_group(df: pd.DataFrame, column: str, group_column: str, values: dict, fallback: float,
                  add_flag: bool = False) -> pd.DataFrame:
    """Fills blanks with the median of the row's group (e.g. age by passenger class); unseen groups use `fallback`."""
    if column not in df.columns:
        return df
    df = df.copy()
    if add_flag:
        df[fresh_name(df, f"{column}_was_missing")] = df[column].isna().astype(int)
    if group_column in df.columns:
        keys = df[group_column].map(lambda v: None if pd.isna(v) else str(v))
        fills = keys.map(lambda k: values.get(k, fallback) if k is not None else fallback).astype(float)
    else:
        fills = pd.Series(fallback, index=df.index, dtype=float)
    df[column] = df[column].fillna(fills)
    return df


def group_rare(df: pd.DataFrame, column: str, keep: list, other_label: str = "Other") -> pd.DataFrame:
    if column not in df.columns:
        return df
    df = df.copy()
    keep_set = set(str(k) for k in keep)
    df[column] = df[column].apply(lambda v: v if pd.isna(v) or str(v) in keep_set else other_label)
    return df


def one_hot(df: pd.DataFrame, column: str, categories: list) -> pd.DataFrame:
    """Creates 0/1 indicator columns for the given categories. Rows where every indicator is 0 belong to the
    baseline category that was deliberately left out (documented in the data dictionary)."""
    if column not in df.columns:
        return df
    as_text = df[column].astype(str)
    df = df.copy()
    for cat, name in zip(categories, indicator_names(df, column, categories)):
        df[name] = (as_text == str(cat)).astype(int)
    return df.drop(columns=[column])


def cast_integer(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Stores whole-number columns (0/1 flags, counts) as integers instead of floats."""
    df = df.copy()
    for c in columns:
        if c in df.columns and df[c].notna().all():
            df[c] = df[c].round().astype(int)
    return df
