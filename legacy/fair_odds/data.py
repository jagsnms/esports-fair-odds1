"""Data loading, validation, and team lookups."""

import csv
import re
import string
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import streamlit as st

from .paths import PROJECT_ROOT

_ZWS = "\u200b\u200c\u200d"


def _clean_spaces(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").replace(_ZWS, "")
    return re.sub(r"\s+", " ", s).strip()


def normalize_name(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("team ", " ").replace("_", " ")
    s = s.translate(str.maketrans("", "", string.punctuation))
    return _clean_spaces(s)


def looks_like(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    if ta and tb and (ta.issubset(tb) or tb.issubset(ta)):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.72


def gosu_name_from_slug(slug: str) -> str:
    s = re.sub(r"^\d+-", "", str(slug or ""))
    return s.replace("-", " ").strip()


def sniff_bad_csv(path: Path, expected_cols: int | None = None, preview_cols: int = 5):
    bad = []
    header = None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.reader(f)
            header = next(rdr)
            exp = expected_cols or len(header)
            for i, row in enumerate(rdr, start=2):
                if len(row) != exp:
                    bad.append((i, len(row), row[:preview_cols]))
                    if len(bad) >= 10:
                        break
            return header, exp, bad
    except Exception as e:
        return header, expected_cols, [("error", str(e), [])]


def read_csv_tolerant(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _coerce_numeric(series, name):
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().any():
        st.warning(
            f"{name}: {int(s.isna().sum())} value(s) could not be parsed; treating as missing."
        )
    return s


def validate_df_cs2(df: pd.DataFrame) -> pd.DataFrame:
    required = ["team", "tier", "rank", "hltv_id", "slug"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CS2 file missing columns: {missing}")
        return df
    df["hltv_id"] = pd.to_numeric(df["hltv_id"], errors="coerce")
    df["tier"] = pd.to_numeric(df["tier"], errors="coerce")
    if df["hltv_id"].isna().any():
        st.info(f"CS2: {int(df['hltv_id'].isna().sum())} team(s) have no hltv_id.")
    if df["tier"].isna().any():
        st.warning(
            f"CS2: {int(df['tier'].isna().sum())} team(s) missing tier; defaulting to Tier 5."
        )
        df["tier"] = df["tier"].fillna(5.0)
    if (df["team"].astype(str).str.strip() == "").any():
        st.warning("CS2: Some rows have empty team names.")
    return df


def validate_df_dota(df: pd.DataFrame) -> pd.DataFrame:
    required = ["team", "tier", "rank", "slug"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Dota file missing columns: {missing}")
        return df.iloc[0:0]
    df["slug"] = df["slug"].fillna(df["team"].astype(str).str.strip().str.replace(" ", "_"))
    df["tier"] = _coerce_numeric(df["tier"], "Dota tier")
    bad = df["tier"].isna() | (df["team"].astype(str).str.strip() == "")
    if bad.any():
        st.warning(f"Dota: dropping {int(bad.sum())} row(s) with missing team/tier.")
        try:
            st.dataframe(
                df.loc[bad, ["team", "tier", "rank", "slug"]], use_container_width=True, height=140
            )
        except Exception:
            pass
    if "opendota_id" not in df.columns:
        df["opendota_id"] = ""
    return df.loc[~bad].copy()


def piecewise_recent_weights(
    n: int, K: int = 6, decay: float = 0.85, floor: float = 0.6, newest_first: bool = True
):
    if n <= 0:
        return []
    idx = range(n) if newest_first else range(n - 1, -1, -1)
    raw = []
    for i in idx:
        if i < K:
            w = 1.0
        else:
            steps = i - K + 1
            w = max(floor, decay**steps)
        raw.append(w)
    s = sum(raw) or 1.0
    factor = n / s
    return [w * factor for w in raw]


def get_team_tier(opp: str, df: pd.DataFrame) -> float:
    target = normalize_name(opp)
    df_local = df.copy()
    if "norm_team" not in df_local.columns:
        df_local["norm_team"] = df_local["team"].apply(normalize_name)
    candidate_cols = ["norm_team"] + (["norm_gosu"] if "norm_gosu" in df_local.columns else [])
    try:
        exact_mask = (df_local[candidate_cols] == target).any(axis=1)
        exact = df_local.loc[exact_mask]
        if not exact.empty:
            return float(exact.iloc[0]["tier"])
    except Exception:
        pass
    best_idx, best_score = None, 0.0
    for idx, row in df_local[candidate_cols].iterrows():
        for cand in row.values:
            cand = str(cand)
            if looks_like(target, cand):
                score = SequenceMatcher(None, target, cand).ratio()
                if score > best_score:
                    best_idx = idx
                    best_score = score
    if best_idx is not None:
        return float(df_local.loc[best_idx, "tier"])
    st.warning(f"No match for '{opp}', defaulting to Tier 5")
    return 5.0


@st.cache_data
def load_cs2_teams() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "cs2_rankings_merged.csv"
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read CS2 CSV: {e}")
        header, exp, bad = sniff_bad_csv(path)
        if bad:
            st.warning(f"Header has {exp} columns. Problem rows: {bad}")
        raise
    df = validate_df_cs2(df)
    df["norm_team"] = df["team"].apply(normalize_name)
    return df


@st.cache_data
def load_dota_teams() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "gosu_dota2_rankings.csv"
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read Dota CSV: {e}")
        header, exp, bad = sniff_bad_csv(path)
        if bad:
            st.warning(f"Header has {exp} columns. Problem rows: {bad}")
        raise
    if "slug" not in df.columns:
        df["slug"] = df["team"].apply(lambda x: str(x).strip().replace(" ", "_"))
    df = validate_df_dota(df)
    df["gosu_display"] = df["slug"].apply(gosu_name_from_slug)
    df["norm_team"] = df["team"].apply(normalize_name)
    df["norm_gosu"] = df["gosu_display"].apply(normalize_name)
    return df
