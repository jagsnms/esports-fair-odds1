# === PART 1/3 ===
# app.py — Esports Fair Odds (CS2 + Dota2)
# Run: streamlit run app.py

import os
import sys
import re
import json
import math
import subprocess
import asyncio
from pathlib import Path
from difflib import SequenceMatcher
import unicodedata
import string
from datetime import datetime
import csv
from typing import Optional  # 3.9-compatible Optional[...] for type hints

import streamlit as st
import pandas as pd
import numpy as np
import requests

# ---- Windows asyncio fix: ensure subprocess support for Playwright ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# ==========================
# App paths (persistent log)
# ==========================
APP_DIR = Path(__file__).resolve().parent
LOG_PATH = APP_DIR / "fair_odds_logs.csv"   # single rolling file

# Canonical schema for logs (prevents mixed old/new columns)
# (Extended to support BO2 3-way fair odds in US.)
LOG_COLUMNS = [
    "timestamp","game","series_format","market_type",
    "team_a","team_b","tier_a","tier_b",
    "K","decay","floor",
    "raw_a","raw_b","adj_a","adj_b","adj_gap",
    "use_logistic","a","b","use_clip","clip_limit",
    # Binary (legacy) model + market + EV
    "p_model","odds_a","odds_b","fair_a","fair_b","ev_a_pct","ev_b_pct",
    # Decision-layer (binary) — retained for CS2/BO3
    "p_decide","ev_a_dec","ev_b_dec","decision","decision_reason",
    "min_edge_pct","prob_gap_pp","shrink_target",
    # --- BO2 3-way fields (new) ---
    "p_map_model","p_map_decide","draw_k",
    "p_a20","p_draw","p_b02",
    "odds_a20","odds_draw","odds_b02",
    "ev_a20_pct","ev_draw_pct","ev_b02_pct",
    "selected_outcome","selected_prob","selected_odds","selected_ev_pct",
    # NEW: fair odds (US) for BO2 outcomes
    "fair_a20_us","fair_draw_us","fair_b02_us",
    # Optional future field (if you add realized outcomes)
    # "result3way"
]

# ==========================
# NEW — Series-score modifier global knobs (safe defaults)
# ==========================
USE_SERIES_SCORE_MOD = True        # default ON; can be toggled in UI
SERIES_CLAMP = 1.25                # absolute per-match cap (same units as base_points_from_tier)
SERIES_WEIGHTS = {0: 0.25, 1: 0.50, 2: 0.75, 3: 1.00, 4: 1.25}  # 5-tier world (abs gap up to 4)
SERIES_PCT_OF_BASE_CAP = 0.40      # cap series bump to ≤ 40% of |base points| for that match

# ==========================
# Helpers (strong normalization + fuzzy match)
# ==========================
_ZWS = "\u200b\u200c\u200d"

def _clean_spaces(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").replace(_ZWS, "")
    return re.sub(r"\s+", " ", s).strip()

def normalize_name(s: str) -> str:
    """Robust canonicalization: NFKC, lowercase, drop 'team ', strip punctuation, collapse spaces."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("team ", " ").replace("_", " ")
    s = s.translate(str.maketrans("", "", string.punctuation))
    return _clean_spaces(s)

def looks_like(a: str, b: str) -> bool:
    """Loose match: token containment OR ~0.72 similarity."""
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    if ta and tb and (ta.issubset(tb) or tb.issubset(ta)):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.72

def gosu_name_from_slug(slug: str) -> str:
    """e.g., '14009-team-spirit' -> 'team spirit'"""
    s = re.sub(r"^\d+-", "", str(slug or ""))
    return s.replace("-", " ").strip()

# ==========================
# CSV Sniffer & tolerant readers (diagnostics)
# ==========================
def sniff_bad_csv(path: Path, expected_cols: Optional[int] = None, preview_cols: int = 5):
    """
    Return (header, expected_cols, bad_rows) where bad_rows is list of tuples:
    (line_number, observed_cols, row_preview_list)
    """
    bad = []
    header = None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.reader(f)
            header = next(rdr)
            exp = expected_cols or len(header)
            for i, row in enumerate(rdr, start=2):  # header is line 1
                if len(row) != exp:
                    bad.append((i, len(row), row[:preview_cols]))
                    if len(bad) >= 10:
                        break
            return header, exp, bad
    except Exception as e:
        return header, expected_cols, [("error", str(e), [])]

def read_csv_tolerant(path: Path) -> pd.DataFrame:
    """Try fast parser; on ParserError, use python engine and skip bad lines."""
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")

# ==========================
# Team CSV validation
# ==========================
def _coerce_numeric(series, name):
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().any():
        st.warning(f"{name}: {int(s.isna().sum())} value(s) could not be parsed; treating as missing.")
    return s

def validate_df_cs2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Do NOT drop rows for missing hltv_id — we still want them for modeling.
    Scraping will be disabled for those teams.
    Missing tier defaults to 5.0.
    """
    required = ["team", "tier", "rank", "hltv_id", "slug"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CS2 file missing columns: {missing}")
        return df  # keep as-is; UI will guard

    # Coerce types but DO NOT DROP
    df["hltv_id"] = pd.to_numeric(df["hltv_id"], errors="coerce")
    df["tier"] = pd.to_numeric(df["tier"], errors="coerce")

    if df["hltv_id"].isna().any():
        st.info(f"CS2: {int(df['hltv_id'].isna().sum())} team(s) have no hltv_id. "
                "Scraping will be disabled for those, but they remain available for calculations.")

    if df["tier"].isna().any():
        st.warning(f"CS2: {int(df['tier'].isna().sum())} team(s) missing tier; defaulting to Tier 5 for those rows.")
        df["tier"] = df["tier"].fillna(5.0)

    if (df["team"].astype(str).str.strip() == "").any():
        st.warning("CS2: Some rows have empty team names; they will appear as blank in selectors.")

    return df

def validate_df_dota(df: pd.DataFrame) -> pd.DataFrame:
    # --- Patched: opendota_id is optional ---
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
            st.dataframe(df.loc[bad, ["team","tier","rank","slug"]], use_container_width=True, height=140)
        except Exception:
            pass

    # ensure column exists for UI code paths that reference it
    if "opendota_id" not in df.columns:
        df["opendota_id"] = ""

    return df.loc[~bad].copy()

# ==========================
# Recency weighting (piecewise, sum-preserving)
# ==========================
def piecewise_recent_weights(n: int, K: int = 6, decay: float = 0.85, floor: float = 0.6, newest_first: bool = True):
    """
    Last K matches weight=1.0; older decay by 'decay' per step, floored at 'floor'.
    Rescale so sum(weights) ~= n (mean weight ≈ 1).
    """
    if n <= 0:
        return []
    idx = range(n) if newest_first else range(n - 1, -1, -1)
    raw = []
    for i in idx:
        if i < K:
            w = 1.0
        else:
            steps = i - K + 1
            w = max(floor, decay ** steps)
        raw.append(w)
    s = sum(raw) or 1.0
    factor = n / s
    return [w * factor for w in raw]

# ==========================
# Tier lookup (Dota-only extras)
# ==========================
def get_team_tier(opp: str, df: pd.DataFrame) -> float:
    """
    Works for both CS2 and Dota:
    - CS2: matches on norm_team only.
    - Dota: if norm_gosu exists, also match on that.
    """
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

# ==========================
# Load CSVs (with robust handling)
# ==========================
@st.cache_data
def load_cs2_teams() -> pd.DataFrame:
    path = APP_DIR / "cs2_rankings_merged.csv"
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read CS2 CSV: {e}")
        header, exp, bad = sniff_bad_csv(path)
        if bad:
            st.warning(f"Header has {exp} columns. Problem rows (line, cols, preview): {bad}")
            st.info("Fix: quote fields containing commas or repair malformed rows.")
        raise
    df = validate_df_cs2(df)
    df["norm_team"] = df["team"].apply(normalize_name)
    return df

@st.cache_data
def load_dota_teams() -> pd.DataFrame:
    path = APP_DIR / "gosu_dota2_rankings.csv"
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read Dota CSV: {e}")
        header, exp, bad = sniff_bad_csv(path)
        if bad:
            st.warning(f"Header has {exp} columns. Problem rows (line, cols, preview): {bad}")
            st.info("Fix: quote fields containing commas or repair malformed rows.")
        raise
    if "slug" not in df.columns:
        df["slug"] = df["team"].apply(lambda x: str(x).strip().replace(" ", "_"))
    df = validate_df_dota(df)
    df["gosu_display"] = df["slug"].apply(gosu_name_from_slug)
    df["norm_team"] = df["team"].apply(normalize_name)
    df["norm_gosu"] = df["gosu_display"].apply(normalize_name)
    return df
# === PART 2/3 ===
# ==========================
# External Scrapers
# ==========================
def scrape_cs2_matches(team_id: str, team_slug: str):
    """Run scraper.py as a subprocess and parse the last JSON line (CS2)."""
    try:
        python_exe = sys.executable
        script_path = str(APP_DIR / "scraper.py")
        result = subprocess.run(
            [python_exe, script_path, str(team_id), team_slug],
            capture_output=True, text=True, timeout=180, cwd=str(APP_DIR),
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()[-1] if result.stdout else "[]"
            return json.loads(line)
        else:
            st.error("CS2 scraper failed")
            with st.expander("CS2 scraper stderr"): st.code(result.stderr or "(no stderr)")
            with st.expander("CS2 scraper stdout"): st.code(result.stdout or "(no stdout)")
            return []
    except Exception as e:
        st.error(f"CS2 scraper error: {e}")
        return []

def scrape_dota_matches_gosu_subprocess(team_slug: str, team_name: str, target: int = 14,
                                        headed: bool = True, browser_channel: str = "bundled",
                                        zoom: int = 80):
    """Run gosu_dota_scraper.py and parse the last JSON array from stdout."""
    try:
        python_exe = sys.executable
        script_path = str(APP_DIR / "gosu_dota_scraper.py")
        cmd = [python_exe, script_path, "--team-slug", team_slug, "--team-name", team_name,
               "--target", str(target), "--zoom", str(zoom)]
        if headed: cmd.append("--headed")
        if browser_channel in ("chrome", "msedge"): cmd += ["--channel", browser_channel]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=str(APP_DIR),
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode != 0:
            st.error(f"Gosu scraper failed (exit {result.returncode}).")
            with st.expander("Gosu scraper stderr"): st.code(result.stderr or "(no stderr)")
            with st.expander("Gosu scraper stdout (tail)"): st.code((result.stdout or "")[-2000:])
            return []

        stdout = result.stdout or ""
        start = stdout.rfind("["); end = stdout.rfind("]")
        if start == -1 or end == -1 or end < start:
            st.warning("Gosu scraper returned no parseable JSON.")
            with st.expander("Gosu scraper raw stdout"): st.code(stdout[-4000:] if stdout else "(empty)")
            return []

        data = json.loads(stdout[start:end+1])
        out = []
        for row in data:
            opp = row.get("opponent", "Unknown")
            win = bool(row.get("win", False))
            # detect draw-ish text
            res_txt = str(row.get("result", "") or row.get("score", "") or row.get("status","")).lower()
            is_draw = ("1-1" in res_txt) or ("draw" in res_txt) or ("tie" in res_txt)
            out.append({"opponent": opp, "win": win, "draw": is_draw})
        return out

    except subprocess.TimeoutExpired:
        st.error("Gosu scraper timed out. Try again or reduce match count.")
        return []
    except Exception as e:
        st.error(f"Gosu scraper error: {e}")
        return []

def fetch_dota_matches(team_id: int, limit: int = 15):
    """OpenDota API fallback."""
    url = f"https://api.opendota.com/api/teams/{team_id}/matches"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            st.error(f"OpenDota API error for team {team_id}: {r.status_code}")
            return []
        data = r.json()
        matches = []
        for m in data[:limit]:
            opp_name = m.get("opposing_team_name", "Unknown")
            radiant = m.get("radiant", None)
            radiant_win = m.get("radiant_win", None)
            if radiant is None or radiant_win is None:
                continue
            win = (radiant and radiant_win) or (not radiant and not radiant_win)
            matches.append({"opponent": opp_name, "win": win, "draw": False})
        return matches
    except Exception as e:
        st.error(f"Error fetching OpenDota data: {e}")
        return []

# ==========================
# Scoring Logic with Tier-Based + Recency Adjustments
# ==========================
def base_points_from_tier(win: bool, tier: float) -> float:
    """Your original tier table as a helper."""
    if win:
        if tier == 1: return 4.0
        elif tier == 1.5: return 3.0
        elif tier == 2: return 2.5
        elif tier == 3: return 2.0
        elif tier == 4: return 1.5
        else: return 1.0
    else:
        if tier == 1: return -1.0
        elif tier == 1.5: return -1.5
        elif tier == 2: return -2.0
        elif tier == 3: return -2.5
        elif tier == 4: return -3.0
        else: return -4.0

# === NEW: helpers to parse series "us/them" into canonical "2-0/2-1/1-2/0-2" ===
def _to_int(x):
    try:
        return int(float(x))
    except Exception:
        return None

def normalize_series_result_from_fields(match: dict) -> str:
    """
    Return a canonical series string from match fields, POV = 'us'.
    Possible returns: "2-0","2-1","1-2","0-2" or "" if unknown/irrelevant.
    Accepted inputs:
      - "series": "2-1"/"1-2"/"2-0"/"0-2"
      - ("us","them") or ("series_us","series_them") or ("maps_us","maps_them") or ("score_us","score_them")
    Draws like 1-1 (BO2) return "" so draw policy handles it.
    """
    s = str(match.get("series", "") or match.get("series_result", "")).strip()
    if s in {"2-0","2-1","1-2","0-2"}:
        return s

    key_pairs = [
        ("us", "them"),
        ("series_us", "series_them"),
        ("maps_us", "maps_them"),
        ("score_us", "score_them"),
    ]
    us = them = None
    for a, b in key_pairs:
        if a in match or b in match:
            us = _to_int(match.get(a))
            them = _to_int(match.get(b))
            break

    if us is None or them is None:
        return ""

    if us == them:
        return ""  # draw-like; handled elsewhere

    if (us, them) == (2, 0): return "2-0"
    if (us, them) == (2, 1): return "2-1"
    if (us, them) == (1, 2): return "1-2"
    if (us, them) == (0, 2): return "0-2"

    # Coerce longer series into Bo3-like buckets from POV 'us'
    return "2-1" if us > them else "1-2"

# === NEW: Series-score modifier (5-tier, Bo3 map counts) ===
def series_score_modifier_5tier(team_tier: float, opp_tier: float, result: str,
                                clamp: float = SERIES_CLAMP,
                                weights: dict = SERIES_WEIGHTS) -> float:
    """
    Series-only, tier-aware bump for Bo3 map counts.
    - result: "2-0","2-1","1-2","0-2" from the POV of the team being scored.
    Rules:
      • No bonus for 0–2 losses (ever).
      • Weaker team rewarded only when they take a map or win.
      • Stronger team penalized for dropping maps/losing.
      • Scales by abs tier gap (0..4). Clamped.
    """
    if result not in {"2-0","2-1","1-2","0-2"}:
        return 0.0

    gap = float(opp_tier) - float(team_tier)  # >0 ⇒ team is stronger (lower numeric tier)
    abs_gap = int(min(4, abs(gap)))
    w = float(weights.get(abs_gap, list(weights.values())[-1]))

    if gap >= 2:  # clearly stronger
        table = {"2-0": 0.0, "2-1": -w, "1-2": -w, "0-2": -w}
        return max(-clamp, min(clamp, table[result]))
    if gap == 1:  # mildly stronger
        table = {"2-0": 0.25, "2-1": -0.50, "1-2": -0.50, "0-2": -0.75}
        return max(-clamp, min(clamp, table[result]))
    if gap == 0:  # peers
        table = {"2-0": 0.25, "2-1": 0.0, "1-2": -0.0, "0-2": -0.25}
        return max(-clamp, min(clamp, table[result]))

    # Weaker side (gap <= -1). No freebies for 0–2.
    if result == "0-2": 
        return 0.0
    if result == "1-2":
        val = 0.50 if abs_gap == 1 else w
        return max(-clamp, min(clamp, val))
    if result in ("2-1","2-0"):
        val = min(w + 0.25, clamp)  # modest upset premium
        return max(-clamp, min(clamp, val))
    return 0.0

def calculate_score(matches,
                    df: pd.DataFrame,
                    current_opponent_tier=None,
                    weight_scheme: str = "piecewise",
                    K: int = 6,
                    decay: float = 0.85,
                    floor: float = 0.6,
                    newest_first: bool = True,
                    # NEW: draw handling
                    draw_policy: str = "graded",   # "loss" | "neutral" | "graded"
                    self_team_tier: Optional[float] = None,
                    draw_gamma: float = 0.5,       # magnitude vs win/loss scale (0..1)
                    draw_gap_cap: float = 3.0,     # cap for tier gap used in scaling
                    draw_gap_power: float = 1.0):  # curve on the gap scaling
    """
    Returns raw_score, adjusted_score, breakdown list.
    - 'graded' draws: positive if opponent stronger, negative if weaker; magnitude grows with tier gap.
    """
    raw_score = 0.0
    adjusted_score = 0.0
    breakdown = []

    n = len(matches)
    if weight_scheme == "piecewise":
        weights = piecewise_recent_weights(n, K=K, decay=decay, floor=floor, newest_first=newest_first)
    else:
        weights = [1.0] * n

    for i, match in enumerate(matches):
        opp = match["opponent"]
        win = bool(match.get("win", False))
        draw_flag = bool(match.get("draw", False))
        tier = get_team_tier(opp, df)

        # --- Base points: win/loss as before; draw depends on policy ---
        if draw_flag:
            if draw_policy == "neutral":
                points = 0.0
                base_txt = "Draw (neutral 0)"
            elif draw_policy == "loss":
                points = base_points_from_tier(False, tier)
                base_txt = f"Draw→Loss rule ({points:+.2f})"
            else:
                # graded
                my_tier = float(self_team_tier) if self_team_tier is not None else 3.0  # fallback
                rel = my_tier - tier
                sign = 1.0 if rel > 0 else (-1.0 if rel < 0 else 0.0)

                win_mag  = base_points_from_tier(True, tier)          # > 0
                loss_mag = abs(base_points_from_tier(False, tier))     # > 0
                base_mag = 0.5 * (win_mag + loss_mag)

                gap = min(abs(rel), draw_gap_cap) / max(draw_gap_cap, 1e-9)
                gap = gap ** max(draw_gap_power, 1e-9)

                points = sign * draw_gamma * base_mag * gap
                base_txt = f"Draw (graded {points:+.2f}; rel_gap={rel:.1f}, base_mag={base_mag:.2f})"
        else:
            # Win/Loss (original table)
            points = base_points_from_tier(True, tier) if win else base_points_from_tier(False, tier)
            base_txt = "Win" if win else "Loss"

            # --- NEW: Series-score bump (uses "series" OR us/them fields if present) ---
            if USE_SERIES_SCORE_MOD:
                series_res = normalize_series_result_from_fields(match)
                if series_res in {"2-0","2-1","1-2","0-2"}:
                    my_tier = float(self_team_tier) if self_team_tier is not None else 3.0
                    s_bump = series_score_modifier_5tier(
                        team_tier=my_tier, opp_tier=tier, result=series_res,
                        clamp=SERIES_CLAMP, weights=SERIES_WEIGHTS
                    )
                    if s_bump:
                        # Percent-of-base cap to keep proportionate
                        pct_cap = SERIES_PCT_OF_BASE_CAP * max(0.5, abs(points))
                        s_bump = max(-pct_cap, min(pct_cap, s_bump))
                        points += s_bump
                        base_txt += f" + SeriesMod({series_res} {s_bump:+.2f})"

        raw_score += points  # audit

        # ---- Tier-gap weighting (compressed) relative to CURRENT opponent tier ----
        if current_opponent_tier is not None:
            tier_gap = tier - current_opponent_tier
            positiveish = win or (draw_flag and draw_policy != "loss" and points >= 0)
            if positiveish:
                if tier_gap < 0:
                    weight_tier_old = 1 + min(0.4, abs(tier_gap) * 0.2)
                else:
                    weight_tier_old = 1 - min(0.3, tier_gap * 0.15)
            else:
                if tier_gap < 0:
                    weight_tier_old = 1 - min(0.2, abs(tier_gap) * 0.1)
                else:
                    weight_tier_old = 1 + min(0.5, tier_gap * 0.25)
            weight_tier_old = max(0.5, min(weight_tier_old, 1.5))
        else:
            tier_gap = 0.0
            weight_tier_old = 1.0

        # Compress toward 1 so final band ~0.85–1.15
        weight_tier = 1 + (weight_tier_old - 1) * 0.3
        weight_tier = max(0.85, min(weight_tier, 1.15))

        # Apply recency on top
        w_match = weights[i]
        adj_points = points * weight_tier * w_match
        adjusted_score += adj_points

        breakdown.append(
            f"{base_txt} vs {opp} (OppTier {tier}, CurOppGap {tier_gap:.1f}) "
            f"Pts={points:+.2f} × TierW={weight_tier:.2f} × RecW={w_match:.3f} = {adj_points:+.3f}"
        )

    return raw_score, adjusted_score, breakdown

# ==========================
# Odds helpers + mapping
# ==========================
def american_to_decimal(odds: int) -> float:
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds)) + 1

def decimal_to_american(o: float) -> int:
    try:
        o = float(o)
    except Exception:
        return 0
    if o <= 1.0:
        return 0
    return int(round((o - 1.0) * 100)) if o >= 2.0 else int(round(-100.0 / (o - 1.0)))

def implied_prob_from_american(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

def calculate_fair_odds_curve(gap: float,
                              base_C=30, alpha=0.03,
                              tail_cutoff=14,
                              L=0.97, k=0.09, x0=0, v=1.0):
    """Original hand-tuned mapping from gap -> p."""
    if abs(gap) <= tail_cutoff:
        C_dynamic = base_C / (1 + alpha * abs(gap))
        p_a = 1 / (1 + 10 ** (-gap / C_dynamic))
    else:
        p_a = L / ((1 + math.exp(-k * (gap - x0))) ** v)
        p_b = L - p_a
        total = p_a + p_b
        p_a /= total
    return p_a

def calculate_fair_odds_from_p(p_a: float):
    p_b = 1 - p_a
    fair_a = -round(100 * p_a / (1 - p_a)) if p_a >= 0.5 else round(100 * (1 - p_a) / p_a)
    fair_b = -round(100 * p_b / (1 - p_b)) if p_b >= 0.5 else round(100 * (1 - p_b) / p_b)
    return fair_a, fair_b

def logistic_mapping(adj_gap: float, a: float, b: float) -> float:
    """p = 1 / (1 + exp(-(a + b*gap)))"""
    return 1.0 / (1.0 + math.exp(-(a + b * adj_gap)))

def color_ev(ev: float) -> str:
    if ev >= 40: return f"✅ **{round(ev, 2)}%**"
    elif 11 <= ev < 40: return f"🟡 {round(ev, 2)}%"
    else: return f"🔴 {round(ev, 2)}%"

# NEW: Probability -> Fair odds (US) helper for BO2 3-way
def prob_to_fair_american(p: float) -> int:
    """Convert a probability to no-vig fair odds (American)."""
    p = max(1e-9, min(1 - 1e-9, float(p)))   # keep in (0,1) to avoid infinities
    dec = 1.0 / p
    return decimal_to_american(dec)

# --- BO2 helpers (new) ---
def compute_bo2_probs(p_map: float, k: float = 1.0):
    """
    From per-map win prob p_map (Team A) produce series-level probs for a BO2:
      A2-0, DRAW(1-1), B0-2.
    'k' scales the draw rate before renormalization (calibration knob).
    """
    p = max(0.0, min(1.0, float(p_map)))
    p_a20 = p * p
    p_draw = 2 * p * (1 - p)
    p_b02 = (1 - p) * (1 - p)
    p_draw *= max(0.0, float(k))
    s = p_a20 + p_draw + p_b02
    if s <= 0:
        return {"A2-0": 0.0, "DRAW": 0.0, "B0-2": 0.0}
    return {"A2-0": p_a20 / s, "DRAW": p_draw / s, "B0-2": p_b02 / s}

def ev_pct_decimal(prob: float, dec_odds: float) -> float:
    """
    EV% for a discrete outcome with probability 'prob' at DECIMAL odds.
    EV% = 100 * (prob * dec_odds - 1)
    """
    if dec_odds is None or dec_odds <= 1.0:
        return float("nan")
    return (prob * dec_odds - 1.0) * 100.0

# ==========================
# Decision layer helper
# ==========================
def decide_bet(p_model: float,
               odds_a: int, odds_b: int,
               n_matches_a: int, n_matches_b: int,
               min_edge_pct: float,
               prob_gap_pp: float,
               shrink_target: int):
    """
    Returns dict: p_decide, ev_a_dec, ev_b_dec, choice ('A'/'B'/None), reason (str).
    Applies:
      - sample-size shrinkage toward 50%
      - model vs market probability gap check (pp)
      - minimum EV threshold (percent)
    """
    eff_matches = min(n_matches_a, n_matches_b)
    lam = min(1.0, eff_matches / float(shrink_target))
    p_decide = lam * p_model + (1.0 - lam) * 0.5

    p_mkt_a = implied_prob_from_american(odds_a)
    gap_ok = abs(p_decide - p_mkt_a) >= (prob_gap_pp / 100.0)

    dec_a = american_to_decimal(odds_a)
    dec_b = american_to_decimal(odds_b)
    ev_a_dec = ((p_decide * dec_a) - 1.0) * 100.0
    ev_b_dec = (((1.0 - p_decide) * dec_b) - 1.0) * 100.0

    reasons = []
    if not gap_ok:
        reasons.append(f"prob gap < {prob_gap_pp}pp")

    best_side = None
    best_ev = max(ev_a_dec, ev_b_dec)
    if best_ev < min_edge_pct:
        reasons.append(f"edge < {min_edge_pct}%")
    else:
        best_side = "A" if ev_a_dec >= ev_b_dec else "B"

    reason = " & ".join(reasons) if reasons else "passes filters"
    return {"p_decide": p_decide, "ev_a_dec": ev_a_dec, "ev_b_dec": ev_b_dec,
            "choice": best_side, "reason": reason}

def decide_bo2_3way(
    p_map_model: float,
    n_matches_a: int, n_matches_b: int,
    min_edge_pct: float,
    prob_gap_pp: float,
    shrink_target: int,
    draw_k: float,
    odds_a20: float, odds_draw: float, odds_b02: float,
):
    """
    3-way decision:
      - shrink p_map toward 0.5 by sample size
      - compute BO2 probs with draw calibration
      - compute EV% per outcome vs DECIMAL odds
      - check prob gap for the selected outcome vs market implied
    Returns dict with selected outcome/EV/prob/odds + reason.
    """
    eff_matches = min(n_matches_a, n_matches_b)
    lam = min(1.0, eff_matches / float(shrink_target))
    p_map_decide = lam * p_map_model + (1.0 - lam) * 0.5

    probs = compute_bo2_probs(p_map_decide, k=draw_k)
    evs = {
        "A2-0": ev_pct_decimal(probs["A2-0"], odds_a20),
        "DRAW": ev_pct_decimal(probs["DRAW"], odds_draw),
        "B0-2": ev_pct_decimal(probs["B0-2"], odds_b02),
    }
    selected = max(evs.items(), key=lambda kv: kv[1])
    sel_outcome, sel_ev = selected[0], selected[1]
    sel_prob = probs[sel_outcome]
    sel_odds = {"A2-0": odds_a20, "DRAW": odds_draw, "B0-2": odds_b02}[sel_outcome]

    imp = {}
    vals = [x for x in [odds_a20, odds_draw, odds_b02] if x and x > 1.0]
    if vals:
        inv = [1.0/x for x in vals]
        s = sum(inv)
        inv_map = {"A2-0": (1.0/odds_a20) if odds_a20 > 1.0 else 0.0,
                   "DRAW": (1.0/odds_draw) if odds_draw > 1.0 else 0.0,
                   "B0-2": (1.0/odds_b02) if odds_b02 > 1.0 else 0.0}
        imp = {k: (v/s if s > 0 else 0.0) for k, v in inv_map.items()}
    p_mkt_sel = imp.get(sel_outcome, 0.0)

    reasons = []
    if sel_ev < min_edge_pct:
        reasons.append(f"edge < {min_edge_pct}%")
    if abs(sel_prob - p_mkt_sel) < (prob_gap_pp / 100.0):
        reasons.append(f"prob gap < {prob_gap_pp}pp")

    ok = (len(reasons) == 0)
    reason = " & ".join(reasons) if reasons else "passes filters"

    return {
        "p_map_decide": p_map_decide,
        "probs": probs,
        "evs": evs,
        "selected_outcome": sel_outcome if ok else None,
        "selected_prob": sel_prob if ok else None,
        "selected_odds": sel_odds if ok else None,
        "selected_ev_pct": sel_ev if ok else None,
        "reason": reason,
    }
# === PART 2/3 (CONT.) ===
# ==========================
# Log schema migration + persistence
# ==========================
def migrate_log_schema(path: Path, out_path: Optional[Path] = None) -> int:
    """
    Reads a mixed-schema log CSV and rewrites it using LOG_COLUMNS.
    Returns number of rows written.
    """
    if out_path is None:
        out_path = path

    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        file_header = next(rdr)
        old_len = len(file_header)
        new_len = len(LOG_COLUMNS)
        for i, row in enumerate(rdr, start=2):
            if len(row) == new_len:
                d = dict(zip(LOG_COLUMNS, row))
            elif len(row) == old_len:
                d = dict(zip(file_header, row))
            else:
                continue
            rows.append({k: d.get(k, "") for k in LOG_COLUMNS})

    seen = set()
    deduped = []
    for d in rows:
        key = (d.get("timestamp",""), d.get("game",""), d.get("team_a",""), d.get("team_b",""), d.get("adj_gap",""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)

    try:
        deduped.sort(key=lambda r: r.get("timestamp",""))
    except Exception:
        pass

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        w.writeheader()
        w.writerows(deduped)

    return len(deduped)

def recompute_metrics_from_logs(rows: list) -> dict:
    """Rebuild Diagnostics counters from logged entries (supports binary & 3-way)."""
    metrics = {"total": 0, "fav_value": 0, "dog_value": 0, "no_bet": 0}
    for r in rows:
        metrics["total"] += 1
        mkt_type = str(r.get("market_type", "")).upper()

        if mkt_type == "3WAY":
            try:
                sel_ev = float(r.get("selected_ev_pct", ""))
                if not (sel_ev == sel_ev):
                    metrics["no_bet"] += 1
                    continue
            except Exception:
                metrics["no_bet"] += 1
                continue

            try:
                oa = float(r.get("odds_a20", 0) or 0)
                od = float(r.get("odds_draw", 0) or 0)
                ob = float(r.get("odds_b02", 0) or 0)
                imp = {}
                vals = [x for x in [oa, od, ob] if x and x > 1.0]
                if vals:
                    inv = [1.0/x for x in vals]; s = sum(inv)
                    inv_map = {"A2-0": (1.0/oa) if oa > 1.0 else 0.0,
                               "DRAW": (1.0/od) if od > 1.0 else 0.0,
                               "B0-2": (1.0/ob) if ob > 1.0 else 0.0}
                    imp = {k: (v/s if s > 0 else 0.0) for k, v in inv_map.items()}
                fav_outcome = max(imp.items(), key=lambda kv: kv[1])[0] if imp else "A2-0"
                sel = str(r.get("selected_outcome",""))
                if sel_ev <= 0:
                    metrics["no_bet"] += 1
                else:
                    if sel == fav_outcome: metrics["fav_value"] += 1
                    else: metrics["dog_value"] += 1
            except Exception:
                metrics["no_bet"] += 1
            continue

        try:
            odds_a = int(float(r.get("odds_a", 0)))
            odds_b = int(float(r.get("odds_b", 0)))
            ev_a = float(r.get("ev_a_dec", r.get("ev_a_pct", -1e9)))
            ev_b = float(r.get("ev_b_dec", r.get("ev_b_pct", -1e9)))
        except Exception:
            metrics["no_bet"] += 1
            continue
        imp_a = implied_prob_from_american(odds_a) if odds_a else 0.0
        imp_b = implied_prob_from_american(odds_b) if odds_b else 0.0
        market_fav = "A" if imp_a >= imp_b else "B"
        if ev_a <= 0 and ev_b <= 0:
            metrics["no_bet"] += 1
        else:
            pick = "A" if ev_a >= ev_b else "B"
            if pick == market_fav: metrics["fav_value"] += 1
            else: metrics["dog_value"] += 1
    return metrics

def load_persisted_logs() -> list:
    """Read LOG_PATH; if malformed/mixed, show diagnostics, recover, and suggest migration."""
    if not LOG_PATH.exists():
        return []
    try:
        df = pd.read_csv(LOG_PATH)
    except pd.errors.ParserError as e:
        st.error(f"Failed to read log CSV ({LOG_PATH.name}): {e}")
        header, exp, bad = sniff_bad_csv(LOG_PATH)
        if bad:
            st.warning(f"Header has {exp} columns. Problem rows (line, cols, preview): {bad}")
            st.info("Click 'Migrate log schema' or 'Repair log file' in Diagnostics to fix this permanently.")
        df = read_csv_tolerant(LOG_PATH)
        st.warning(f"Loaded log after skipping malformed rows. Rows kept: {len(df)}")
    except Exception as e:
        st.error(f"Error reading log CSV: {e}")
        return []
    for c in LOG_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[LOG_COLUMNS].to_dict(orient="records")

def persist_log_row(entry: dict):
    """Append one row to disk in canonical order (create file with header if not present)."""
    row = {k: entry.get(k, "") for k in LOG_COLUMNS}
    df = pd.DataFrame([row], columns=LOG_COLUMNS)
    write_header = not LOG_PATH.exists()
    df.to_csv(LOG_PATH, mode="a", header=write_header, index=False, line_terminator="\n")

# ==========================
# Session diagnostics & logging
# ==========================
def init_metrics():
    if "logs" not in st.session_state:
        st.session_state["logs"] = load_persisted_logs()
    st.session_state["metrics"] = recompute_metrics_from_logs(st.session_state["logs"])

def update_metrics_binary(ev_a: float, ev_b: float, odds_a: int, odds_b: int):
    m = st.session_state.get("metrics", {"total": 0, "fav_value": 0, "dog_value": 0, "no_bet": 0})
    m["total"] += 1
    imp_a = implied_prob_from_american(odds_a) if odds_a else 0.0
    imp_b = implied_prob_from_american(odds_b) if odds_b else 0.0
    market_fav = "A" if imp_a >= imp_b else "B"
    if (ev_a is not None and ev_a > 0) or (ev_b is not None and ev_b > 0):
        pick = "A" if (ev_a or -1e9) >= (ev_b or -1e9) else "B"
        best_ev = ev_a if pick == "A" else ev_b
        if best_ev is None or best_ev <= 0:
            m["no_bet"] += 1
        else:
            if pick == market_fav: m["fav_value"] += 1
            else: m["dog_value"] += 1
    else:
        m["no_bet"] += 1
    st.session_state["metrics"] = m

def update_metrics_3way(selected_ev_pct: Optional[float], selected_outcome: Optional[str],
                        odds_a20: float, odds_draw: float, odds_b02: float):
    m = st.session_state.get("metrics", {"total": 0, "fav_value": 0, "dog_value": 0, "no_bet": 0})
    m["total"] += 1
    if not selected_outcome or selected_ev_pct is None or selected_ev_pct <= 0:
        m["no_bet"] += 1
    else:
        oa, od, ob = odds_a20, odds_draw, odds_b02
        inv_map = {"A2-0": (1.0/oa) if oa and oa > 1 else 0.0,
                   "DRAW": (1.0/od) if od and od > 1 else 0.0,
                   "B0-2": (1.0/ob) if ob and ob > 1 else 0.0}
        s = sum(inv_map.values()) or 1.0
        imp = {k: v/s for k, v in inv_map.items()}
        fav_outcome = max(imp.items(), key=lambda kv: kv[1])[0]
        if selected_outcome == fav_outcome: m["fav_value"] += 1
        else: m["dog_value"] += 1
    st.session_state["metrics"] = m

def log_row(entry: dict):
    st.session_state["logs"].append({k: entry.get(k, "") for k in LOG_COLUMNS})
    try:
        persist_log_row(entry)
    except Exception as e:
        st.warning(f"Could not persist log row: {e}")
    st.session_state["metrics"] = recompute_metrics_from_logs(st.session_state["logs"])

def export_logs_df() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.get("logs", []), columns=LOG_COLUMNS)
# === PART 3/3 ===
# ==========================
# Streamlit App
# ==========================
st.title("Esports Fair Odds Calculator (CS2 + Dota2)")
init_metrics()

# Global calibration + decision controls (applies to both tabs)
with st.expander("Calibration & Mapping (optional)"):
    use_clip = st.checkbox("Clip extreme adjusted gaps before mapping", value=False,
                           help="Prevents absurd probabilities on outlier gaps.")
    clip_limit = st.slider("Clip limit (|gap|)", 10, 40, 25, 1)
    use_logistic = st.checkbox("Use calibrated logistic mapping (gap → p)", value=False,
                               help="OFF = original curve. ON = p = 1/(1+exp(-(a+b*gap))).")
    colA, colB = st.columns(2)
    with colA:
        a_param = st.number_input("Logistic a (intercept)",
                                  value=st.session_state.get("a_param", 0.0),
                                  step=0.01, format="%.4f")
    with colB:
        b_param = st.number_input("Logistic b (slope)",
                                  value=st.session_state.get("b_param", 0.18),
                                  step=0.01, format="%.4f",
                                  help="Fit this from your logged data later. Placeholder default.")

    st.markdown("---")
    # --- Decision layer controls ---
    min_edge_pct = st.slider(
        "Minimum EV to bet (%)", 0, 15, 5, 1,
        help="Edges below this are auto 'No bet' for logging/decision. Model EVs remain unchanged."
    )
    prob_gap_pp = st.slider(
        "Minimum model vs market probability gap (percentage points)", 0, 5, 3, 1,
        help="Require |p_model - p_market| ≥ this many percentage points to consider a bet."
    )
    shrink_target_matches = st.slider(
        "Effective matches for full confidence (shrinkage target)", 6, 20, 12, 1,
        help="Blends model p toward 50% when data is thin. Set higher = stricter."
    )

    st.markdown("---")
    # --- Series modifier controls (UI override of globals) ---
    use_series_mod_ui = st.checkbox(
        "Use series score modifier (map-count aware, tier-adjusted)",
        value=True,
        help="Rewards weaker teams when they take maps/wins; penalizes stronger teams for dropping maps. No bonuses for 0–2 losses."
    )
    series_clamp_ui = st.slider("Series modifier absolute clamp (points)", 0.25, 2.00, float(SERIES_CLAMP), 0.05)
    series_pct_cap_ui = st.slider("Series modifier cap as % of base", 0.10, 0.60, float(SERIES_PCT_OF_BASE_CAP), 0.05)

    # apply UI overrides to globals
    USE_SERIES_SCORE_MOD = bool(use_series_mod_ui)
    SERIES_CLAMP = float(series_clamp_ui)
    SERIES_PCT_OF_BASE_CAP = float(series_pct_cap_ui)

tabs = st.tabs(["CS2", "Dota2", "Diagnostics / Export", "CS2 In-Play Indicator (MVP)"])

# --------------------------
# CS2 TAB
# --------------------------
with tabs[0]:
    st.header("CS2 Fair Odds")
    df_cs2 = load_cs2_teams()

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A (CS2)", df_cs2["team"].tolist(), key="cs2_a")
        row_a = df_cs2.loc[df_cs2["team"] == team_a].iloc[0]
        team_a_slug = row_a["slug"]
        hltv_a_val = row_a["hltv_id"]
        team_a_id = str(int(hltv_a_val)) if pd.notna(hltv_a_val) else None

        if team_a_id is None:
            st.caption("Team A: no hltv_id — scraping disabled (calculations still work).")

        if st.button("Scrape Team A (CS2)"):
            if team_a_id is None or not isinstance(team_a_slug, str) or not team_a_slug:
                st.error("Cannot scrape Team A: missing hltv_id or slug for this team.")
                st.info("You can still Calculate using the tiers; scraping just won’t work for this team.")
                matches_a = []
            else:
                matches_a = scrape_cs2_matches(team_a_id, team_a_slug)
            st.session_state["cs2_matches_a"] = matches_a
        matches_a = st.session_state.get("cs2_matches_a", [])
        st.write(matches_a)

    with col2:
        team_b = st.selectbox("Select Team B (CS2)", df_cs2["team"].tolist(), key="cs2_b")
        row_b = df_cs2.loc[df_cs2["team"] == team_b].iloc[0]
        team_b_slug = row_b["slug"]
        hltv_b_val = row_b["hltv_id"]
        team_b_id = str(int(hltv_b_val)) if pd.notna(hltv_b_val) else None

        if team_b_id is None:
            st.caption("Team B: no hltv_id — scraping disabled (calculations still work).")

        if st.button("Scrape Team B (CS2)"):
            if team_b_id is None or not isinstance(team_b_slug, str) or not team_b_slug:
                st.error("Cannot scrape Team B: missing hltv_id or slug for this team.")
                st.info("You can still Calculate using the tiers; scraping just won’t work for this team.")
                matches_b = []
            else:
                matches_b = scrape_cs2_matches(team_b_id, team_b_slug)
            st.session_state["cs2_matches_b"] = matches_b
        matches_b = st.session_state.get("cs2_matches_b", [])
        st.write(matches_b)

    odds_a = st.number_input("Team A Market Odds (CS2) — American", value=-140, key="cs2_odds_a")
    odds_b = st.number_input("Team B Market Odds (CS2) — American", value=+120, key="cs2_odds_b")

    st.subheader("Recency Weighting (CS2)")
    K_cs2 = st.slider("Full-weight recent matches (K)", 3, 12, 6, key="K_cs2")
    decay_cs2 = st.slider("Decay per step beyond K", 0.75, 0.95, 0.85, 0.01, key="decay_cs2")
    floor_cs2 = st.slider("Minimum weight floor", 0.40, 0.90, 0.60, 0.01, key="floor_cs2")

    if st.button("Calculate (CS2)"):
        if matches_a and matches_b:
            min_matches = min(len(matches_a), len(matches_b))
            matches_a = matches_a[:min_matches]
            matches_b = matches_b[:min_matches]
            st.text(f"Using last {min_matches} matches for both teams.")

            team_a_tier = float(row_a["tier"])
            team_b_tier = float(row_b["tier"])

            raw_a, adj_a, breakdown_a = calculate_score(
                matches_a, df_cs2, current_opponent_tier=team_b_tier,
                weight_scheme="piecewise", K=K_cs2, decay=decay_cs2, floor=floor_cs2, newest_first=True,
                draw_policy="loss", self_team_tier=team_a_tier
            )
            raw_b, adj_b, breakdown_b = calculate_score(
                matches_b, df_cs2, current_opponent_tier=team_a_tier,
                weight_scheme="piecewise", K=K_cs2, decay=decay_cs2, floor=floor_cs2, newest_first=True,
                draw_policy="loss", self_team_tier=team_b_tier
            )

            raw_gap = raw_a - raw_b
            adj_gap = adj_a - adj_b
            if use_clip:
                adj_gap = max(min(adj_gap, clip_limit), -clip_limit)

            p = logistic_mapping(adj_gap, a_param, b_param) if use_logistic else calculate_fair_odds_curve(adj_gap)
            dec_a = american_to_decimal(odds_a); dec_b = american_to_decimal(odds_b)
            ev_a = ((p * dec_a) - 1) * 100
            ev_b = (((1 - p) * dec_b) - 1) * 100
            fair_a, fair_b = calculate_fair_odds_from_p(p)

            st.subheader("Summary (CS2)")
            st.text(f"{team_a} (Tier {team_a_tier}) vs {team_b} (Tier {team_b_tier})")
            st.text(f"Raw Gap: {raw_gap:.2f}, Adjusted Gap: {adj_gap:.2f}")
            st.text(f"Market Odds: {odds_a} / {odds_b}")
            st.text(f"Fair Odds:   {fair_a} / {fair_b}")
            st.text(f"Win Probability: {round(p * 100, 2)}%")
            st.caption("EV below reflects raw model p (pre-decision filters).")
            st.markdown(f"EV: {color_ev(ev_a)} / {color_ev(ev_b)}")
            st.text(f"Score Breakdown: Raw {raw_a:.2f}/{raw_b:.2f} | Adj {adj_a:.3f}/{adj_b:.3f}")

            st.subheader(f"{team_a} Breakdown")
            for line in breakdown_a: st.text(line)
            st.subheader(f"{team_b} Breakdown")
            for line in breakdown_b: st.text(line)

            decision = decide_bet(
                p_model=p,
                odds_a=odds_a, odds_b=odds_b,
                n_matches_a=len(matches_a), n_matches_b=len(matches_b),
                min_edge_pct=min_edge_pct,
                prob_gap_pp=prob_gap_pp,
                shrink_target=shrink_target_matches,
            )
            if decision["choice"] is None:
                st.markdown(f"**Decision:** ❌ No bet** — {decision['reason']}")
                ev_a_eff = ev_b_eff = None
            else:
                pick_team = team_a if decision["choice"] == "A" else team_b
                pick_ev = decision["ev_a_dec"] if decision["choice"] == "A" else decision["ev_b_dec"]
                st.markdown(f"**Decision:** ✅ Bet **{pick_team}** ({pick_ev:+.2f}% EV) — {decision['reason']}")
                ev_a_eff = decision["ev_a_dec"] if decision["choice"] == "A" else None
                ev_b_eff = decision["ev_b_dec"] if decision["choice"] == "B" else None

            update_metrics_binary(ev_a_eff, ev_b_eff, odds_a, odds_b)
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "game": "CS2",
                "series_format": "BO3",
                "market_type": "BINARY",
                "team_a": team_a, "team_b": team_b,
                "tier_a": team_a_tier, "tier_b": team_b_tier,
                "K": K_cs2, "decay": decay_cs2, "floor": floor_cs2,
                "raw_a": raw_a, "raw_b": raw_b,
                "adj_a": adj_a, "adj_b": adj_b, "adj_gap": adj_gap,
                "use_logistic": use_logistic, "a": a_param, "b": b_param,
                "use_clip": use_clip, "clip_limit": clip_limit,
                "p_model": p,
                "odds_a": odds_a, "odds_b": odds_b,
                "fair_a": fair_a, "fair_b": fair_b,
                "ev_a_pct": ev_a, "ev_b_pct": ev_b,
                "p_decide": decision["p_decide"],
                "ev_a_dec": decision["ev_a_dec"],
                "ev_b_dec": decision["ev_b_dec"],
                "decision": decision["choice"] or "NO_BET",
                "decision_reason": decision["reason"],
                "min_edge_pct": min_edge_pct,
                "prob_gap_pp": prob_gap_pp,
                "shrink_target": shrink_target_matches,
                "p_map_model":"", "p_map_decide":"", "draw_k":"",
                "p_a20":"", "p_draw":"", "p_b02":"",
                "odds_a20":"", "odds_draw":"", "odds_b02":"",
                "ev_a20_pct":"", "ev_draw_pct":"", "ev_b02_pct":"",
                "selected_outcome":"", "selected_prob":"", "selected_odds":"", "selected_ev_pct":"",
                "fair_a20_us":"", "fair_draw_us":"", "fair_b02_us":""
            }
            log_row(entry)
        else:
            st.warning("Scrape both teams first.")

# --------------------------
# Dota TAB  (with BO2 3-way + graded draws)
# --------------------------
with tabs[1]:
    st.header("Dota 2 Fair Odds")
    df_dota = load_dota_teams()
    dota_ok = df_dota

    series_format = st.radio(
        "Series format",
        options=["BO3 (binary win/lose)", "BO2 (3-way with draw)"],
        horizontal=True,
        index=0,
        help="BO2 adds a proper 3-way (A 2–0 / Draw 1–1 / B 0–2) market."
    )

    src = st.radio("Data source", ["OpenDota (API)", "GosuGamers (Scrape)"], horizontal=True, index=1)
    headed_toggle = st.toggle("Show browser during Gosu scrape", value=True,
                              help="Use a visible browser window for the scrape.")
    browser_channel = st.selectbox("Browser for Gosu scrape",
        options=["bundled", "chrome", "msedge"], index=0, key="dota_browser_channel",
        help="Use 'bundled' to mimic CLI (Playwright Chromium). Or force Chrome/Edge if installed.",
    )
    zoom_pct = st.slider("Gosu page zoom (%)", min_value=60, max_value=110, value=80, step=5,
                         help="Zoom out to keep the paginator in view.")
    target_matches = st.slider("Matches to use (last N)", min_value=8, max_value=30, value=14, step=1)

    st.subheader("Recency Weighting (Dota2)")
    K_dota = st.slider("Full-weight recent matches (K)", 3, 12, 6, key="K_dota")
    decay_dota = st.slider("Decay per step beyond K", 0.75, 0.95, 0.85, 0.01, key="decay_dota")
    floor_dota = st.slider("Minimum weight floor", 0.40, 0.90, 0.60, 0.01, key="floor_dota")

    st.subheader("Bo2 Draw Handling")
    draw_mode = st.radio(
        "Draw policy",
        ["Graded by tier (recommended)", "Neutral (0 points)", "Legacy (treat as loss)"],
        horizontal=False,
        index=0,
        help="Graded: + if draw vs stronger, − if draw vs weaker; scaled by tier gap."
    )
    draw_gamma = st.slider("Draw magnitude γ (0–1)", 0.00, 1.00, 0.50, 0.05,
                           help="0 = ignore draws, 1 = half-way to a full win/loss at max gap.")
    draw_gap_cap = st.slider("Tier gap cap for draws", 1.0, 4.0, 3.0, 0.5,
                             help="How many tier steps until a draw is 'maximally' graded.")
    draw_gap_power = st.slider("Tier gap curve (power)", 0.5, 2.0, 1.0, 0.1,
                               help="<1 boosts small gaps; >1 emphasizes big gaps.")
    draw_policy = "graded" if draw_mode.startswith("Graded") else ("neutral" if draw_mode.startswith("Neutral") else "loss")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A (Dota2)", dota_ok["team"].tolist(), key="dota_a")
        row_a = dota_ok.loc[dota_ok["team"] == team_a].iloc[0]
        team_a_slug = row_a["slug"]

        team_a_id = None
        try:
            if "opendota_id" in row_a and pd.notna(row_a["opendota_id"]) and str(row_a["opendota_id"]).strip() != "":
                team_a_id = int(float(row_a["opendota_id"]))
        except Exception:
            team_a_id = None

        if st.button("Scrape Team A (Dota2)"):
            if src.startswith("OpenDota"):
                if team_a_id is None:
                    st.info("No OpenDota ID for this team — falling back to Gosu scraping.")
                    gosu_name_a = gosu_name_from_slug(team_a_slug)
                    matches_a = scrape_dota_matches_gosu_subprocess(
                        team_slug=team_a_slug, team_name=gosu_name_a,
                        target=target_matches, headed=headed_toggle,
                        browser_channel=browser_channel, zoom=zoom_pct,
                    )
                else:
                    matches_a = fetch_dota_matches(team_a_id, limit=target_matches)
            else:
                gosu_name_a = gosu_name_from_slug(team_a_slug)
                matches_a = scrape_dota_matches_gosu_subprocess(
                    team_slug=team_a_slug, team_name=gosu_name_a,
                    target=target_matches, headed=headed_toggle,
                    browser_channel=browser_channel, zoom=zoom_pct,
                )
            st.session_state["dota_matches_a"] = matches_a
        matches_a = st.session_state.get("dota_matches_a", [])
        st.write(matches_a)

    with col2:
        team_b = st.selectbox("Select Team B (Dota2)", dota_ok["team"].tolist(), key="dota_b")
        row_b = dota_ok.loc[dota_ok["team"] == team_b].iloc[0]
        team_b_slug = row_b["slug"]

        team_b_id = None
        try:
            if "opendota_id" in row_b and pd.notna(row_b["opendota_id"]) and str(row_b["opendota_id"]).strip() != "":
                team_b_id = int(float(row_b["opendota_id"]))
        except Exception:
            team_b_id = None

        if st.button("Scrape Team B (Dota2)"):
            if src.startswith("OpenDota"):
                if team_b_id is None:
                    st.info("No OpenDota ID for this team — falling back to Gosu scraping.")
                    gosu_name_b = gosu_name_from_slug(team_b_slug)
                    matches_b = scrape_dota_matches_gosu_subprocess(
                        team_slug=team_b_slug, team_name=gosu_name_b,
                        target=target_matches, headed=headed_toggle,
                        browser_channel=browser_channel, zoom=zoom_pct,
                    )
                else:
                    matches_b = fetch_dota_matches(team_b_id, limit=target_matches)
            else:
                gosu_name_b = gosu_name_from_slug(team_b_slug)
                matches_b = scrape_dota_matches_gosu_subprocess(
                    team_slug=team_b_slug, team_name=gosu_name_b,
                    target=target_matches, headed=headed_toggle,
                    browser_channel=browser_channel, zoom=zoom_pct,
                )
            st.session_state["dota_matches_b"] = matches_b
        matches_b = st.session_state.get("dota_matches_b", [])
        st.write(matches_b)

    if series_format.startswith("BO3"):
        odds_a = st.number_input("Team A Market Odds (BO3) — American", value=-140, key="dota_odds_a")
        odds_b = st.number_input("Team B Market Odds (BO3) — American", value=+120, key="dota_odds_b")
    else:
        st.markdown("### BO2 — 3-Way Market (A 2–0 / Draw 1–1 / B 0–2)")
        draw_k = st.slider("Draw calibration k (fit later; 1.0 = neutral)", 0.50, 1.50, 1.00, 0.01)

        colA2, colD, colB2 = st.columns(3)
        with colA2:
            odds_a20_us = st.number_input("Odds: A 2–0 (American)", value=+220, step=1, format="%d")
        with colD:
            odds_draw_us = st.number_input("Odds: Draw 1–1 (American)", value=+110, step=1, format="%d")
        with colB2:
            odds_b02_us = st.number_input("Odds: B 0–2 (American)", value=+260, step=1, format="%d")

        odds_a20 = american_to_decimal(int(odds_a20_us))
        odds_draw = american_to_decimal(int(odds_draw_us))
        odds_b02 = american_to_decimal(int(odds_b02_us))

    if st.button(f"Calculate (Dota2 — {series_format.split()[0]})"):
        if "dota_matches_a" in st.session_state and "dota_matches_b" in st.session_state:
            matches_a = st.session_state["dota_matches_a"][:target_matches]
            matches_b = st.session_state["dota_matches_b"][:target_matches]
            if not matches_a or not matches_b:
                st.warning("Scrape both teams first (Dota2).")
            else:
                st.text(f"Using last {min(len(matches_a), len(matches_b))} matches for both teams.")

                team_a_tier = float(row_a["tier"])
                team_b_tier = float(row_b["tier"])

                raw_a, adj_a, breakdown_a = calculate_score(
                    matches_a, df_dota, current_opponent_tier=team_b_tier,
                    weight_scheme="piecewise", K=K_dota, decay=decay_dota, floor=floor_dota, newest_first=True,
                    draw_policy=draw_policy, self_team_tier=team_a_tier,
                    draw_gamma=draw_gamma, draw_gap_cap=draw_gap_cap, draw_gap_power=draw_gap_power
                )
                raw_b, adj_b, breakdown_b = calculate_score(
                    matches_b, df_dota, current_opponent_tier=team_a_tier,
                    weight_scheme="piecewise", K=K_dota, decay=decay_dota, floor=floor_dota, newest_first=True,
                    draw_policy=draw_policy, self_team_tier=team_b_tier,
                    draw_gamma=draw_gamma, draw_gap_cap=draw_gap_cap, draw_gap_power=draw_gap_power
                )

                raw_gap = raw_a - raw_b
                adj_gap = adj_a - adj_b
                if use_clip:
                    adj_gap = max(min(adj_gap, clip_limit), -clip_limit)

                p_map = logistic_mapping(adj_gap, a_param, b_param) if use_logistic else calculate_fair_odds_curve(adj_gap)

                if series_format.startswith("BO3"):
                    dec_a = american_to_decimal(odds_a); dec_b = american_to_decimal(odds_b)
                    ev_a = ((p_map * dec_a) - 1) * 100
                    ev_b = (((1 - p_map) * dec_b) - 1) * 100
                    fair_a, fair_b = calculate_fair_odds_from_p(p_map)

                    st.subheader("Summary (Dota2 — BO3)")
                    st.text(f"{team_a} (Tier {team_a_tier}) vs {team_b} (Tier {team_b_tier})")
                    st.text(f"Raw Gap: {raw_gap:.2f}, Adjusted Gap: {adj_gap:.2f}")
                    st.text(f"Market Odds: {odds_a} / {odds_b}")
                    st.text(f"Fair Odds:   {fair_a} / {fair_b}")
                    st.text(f"Win Probability (series): {round(p_map * 100, 2)}%")
                    st.caption("EV below reflects raw model p (pre-decision filters).")
                    st.markdown(f"EV: {color_ev(ev_a)} / {color_ev(ev_b)}")
                    st.text(f"Score Breakdown: Raw {raw_a:.2f}/{raw_b:.2f} | Adj {adj_a:.3f}/{adj_b:.3f}")

                    st.subheader(f"{team_a} Breakdown")
                    for line in breakdown_a: st.text(line)
                    st.subheader(f"{team_b} Breakdown")
                    for line in breakdown_b: st.text(line)

                    decision = decide_bet(
                        p_model=p_map,
                        odds_a=odds_a, odds_b=odds_b,
                        n_matches_a=len(matches_a), n_matches_b=len(matches_b),
                        min_edge_pct=min_edge_pct,
                        prob_gap_pp=prob_gap_pp,
                        shrink_target=shrink_target_matches,
                    )
                    if decision["choice"] is None:
                        st.markdown(f"**Decision:** ❌ No bet** — {decision['reason']}")
                        ev_a_eff = ev_b_eff = None
                    else:
                        pick_team = team_a if decision["choice"] == "A" else team_b
                        pick_ev = decision["ev_a_dec"] if decision["choice"] == "A" else decision["ev_b_dec"]
                        st.markdown(f"**Decision:** ✅ Bet **{pick_team}** ({pick_ev:+.2f}% EV) — {decision['reason']}")
                        ev_a_eff = decision["ev_a_dec"] if decision["choice"] == "A" else None
                        ev_b_eff = decision["ev_b_dec"] if decision["choice"] == "B" else None

                    update_metrics_binary(ev_a_eff, ev_b_eff, odds_a, odds_b)
                    entry = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "game": "Dota2",
                        "series_format": "BO3",
                        "market_type": "BINARY",
                        "team_a": team_a, "team_b": team_b,
                        "tier_a": team_a_tier, "tier_b": team_b_tier,
                        "K": K_dota, "decay": decay_dota, "floor": floor_dota,
                        "raw_a": raw_a, "raw_b": raw_b,
                        "adj_a": adj_a, "adj_b": adj_b, "adj_gap": adj_gap,
                        "use_logistic": use_logistic, "a": a_param, "b": b_param,
                        "use_clip": use_clip, "clip_limit": clip_limit,
                        "p_model": p_map,
                        "odds_a": odds_a, "odds_b": odds_b,
                        "fair_a": fair_a, "fair_b": fair_b,
                        "ev_a_pct": ev_a, "ev_b_pct": ev_b,
                        "p_decide": decision["p_decide"],
                        "ev_a_dec": decision["ev_a_dec"],
                        "ev_b_dec": decision["ev_b_dec"],
                        "decision": decision["choice"] or "NO_BET",
                        "decision_reason": decision["reason"],
                        "min_edge_pct": min_edge_pct,
                        "prob_gap_pp": prob_gap_pp,
                        "shrink_target": shrink_target_matches,
                        "p_map_model":"", "p_map_decide":"", "draw_k":"",
                        "p_a20":"", "p_draw":"", "p_b02":"",
                        "odds_a20":"", "odds_draw":"", "odds_b02":"",
                        "ev_a20_pct":"", "ev_draw_pct":"", "ev_b02_pct":"",
                        "selected_outcome":"", "selected_prob":"", "selected_odds":"", "selected_ev_pct":"",
                        "fair_a20_us":"", "fair_draw_us":"", "fair_b02_us":""
                    }
                    log_row(entry)

                else:
                    st.subheader("Summary (Dota2 — BO2 3-Way)")
                    st.text(f"{team_a} (Tier {team_a_tier}) vs {team_b} (Tier {team_b_tier})")
                    st.text(f"Raw Gap: {raw_gap:.2f}, Adjusted Gap: {adj_gap:.2f}")
                    st.text(f"Per-map win prob (Team A) — model: {round(p_map*100,2)}%")

                    probs_model = compute_bo2_probs(p_map, k=draw_k)
                    evs_model = {
                        "A2-0": ev_pct_decimal(probs_model["A2-0"], odds_a20),
                        "DRAW": ev_pct_decimal(probs_model["DRAW"], odds_draw),
                        "B0-2": ev_pct_decimal(probs_model["B0-2"], odds_b02),
                    }
                    fair_a20_us  = prob_to_fair_american(probs_model["A2-0"])
                    fair_draw_us = prob_to_fair_american(probs_model["DRAW"])
                    fair_b02_us  = prob_to_fair_american(probs_model["B0-2"])

                    df_preview = pd.DataFrame([
                        {"Outcome":"A 2–0","Prob%":round(probs_model["A2-0"]*100,2),"Odds (US)": int(odds_a20_us),"Fair (US)": int(fair_a20_us),"EV%":round(evs_model["A2-0"],2)},
                        {"Outcome":"Draw 1–1","Prob%":round(probs_model["DRAW"]*100,2),"Odds (US)": int(odds_draw_us),"Fair (US)": int(fair_draw_us),"EV%":round(evs_model["DRAW"],2)},
                        {"Outcome":"B 0–2","Prob%":round(probs_model["B0-2"]*100,2),"Odds (US)": int(odds_b02_us),"Fair (US)": int(fair_b02_us),"EV%":round(evs_model["B0-2"],2)},
                    ])
                    st.dataframe(df_preview, use_container_width=True)
                    st.text(f"Fair (US): A2–0 {fair_a20_us:+d} | Draw {fair_draw_us:+d} | B0–2 {fair_b02_us:+d}")

                    dec3 = decide_bo2_3way(
                        p_map_model=p_map,
                        n_matches_a=len(matches_a), n_matches_b=len(matches_b),
                        min_edge_pct=min_edge_pct,
                        prob_gap_pp=prob_gap_pp,
                        shrink_target=shrink_target_matches,
                        draw_k=draw_k,
                        odds_a20=odds_a20, odds_draw=odds_draw, odds_b02=odds_b02,
                    )

                    if dec3["selected_outcome"] is None:
                        st.markdown(f"**Decision:** ❌ No bet** — {dec3['reason']}")
                        update_metrics_3way(None, None, odds_a20, odds_draw, odds_b02)
                    else:
                        sel_us = decimal_to_american(dec3["selected_odds"])
                        st.markdown(
                            f"**Decision:** ✅ Bet **{dec3['selected_outcome']}** "
                            f"(EV {dec3['selected_ev_pct']:+.2f}%, Prob {dec3['selected_prob']*100:.2f}%, "
                            f"Odds {sel_us:+d}) — {dec3['reason']}"
                        )
                        update_metrics_3way(dec3["selected_ev_pct"], dec3["selected_outcome"],
                                            odds_a20, odds_draw, odds_b02)

                    entry = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "game": "Dota2",
                        "series_format": "BO2",
                        "market_type": "3WAY",
                        "team_a": team_a, "team_b": team_b,
                        "tier_a": team_a_tier, "tier_b": team_b_tier,
                        "K": K_dota, "decay": decay_dota, "floor": floor_dota,
                        "raw_a": raw_a, "raw_b": raw_b,
                        "adj_a": adj_a, "adj_b": adj_b, "adj_gap": adj_gap,
                        "use_logistic": use_logistic, "a": a_param, "b": b_param,
                        "use_clip": use_clip, "clip_limit": clip_limit,
                        "p_model": "", "odds_a": "", "odds_b": "", "fair_a": "", "fair_b": "", "ev_a_pct": "", "ev_b_pct": "",
                        "p_decide": "", "ev_a_dec": "", "ev_b_dec": "",
                        "decision": (dec3["selected_outcome"] or "NO_BET"),
                        "decision_reason": dec3["reason"],
                        "min_edge_pct": min_edge_pct,
                        "prob_gap_pp": prob_gap_pp,
                        "shrink_target": shrink_target_matches,
                        "p_map_model": p_map,
                        "p_map_decide": dec3["p_map_decide"],
                        "draw_k": draw_k,
                        "p_a20": probs_model["A2-0"], "p_draw": probs_model["DRAW"], "p_b02": probs_model["B0-2"],
                        "odds_a20": odds_a20, "odds_draw": odds_draw, "odds_b02": odds_b02,
                        "ev_a20_pct": evs_model["A2-0"], "ev_draw_pct": evs_model["DRAW"], "ev_b02_pct": evs_model["B0-2"],
                        "selected_outcome": dec3["selected_outcome"] or "",
                        "selected_prob": dec3["selected_prob"] or "",
                        "selected_odds": dec3["selected_odds"] or "",
                        "selected_ev_pct": dec3["selected_ev_pct"] or "",
                        "fair_a20_us": fair_a20_us, "fair_draw_us": fair_draw_us, "fair_b02_us": fair_b02_us,
                    }
                    log_row(entry)
        else:
            st.warning("Scrape both teams first (Dota2).")

# --------------------------
# Diagnostics / Export
# --------------------------
with tabs[2]:
    st.header("Diagnostics")
    m = st.session_state["metrics"]
    total = max(1, m["total"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Calcs", m["total"])
    col2.metric("Fav value %", f"{(m['fav_value']/total)*100:.1f}%")
    col3.metric("Dog value %", f"{(m['dog_value']/total)*100:.1f}%")
    col4.metric("No-bet %", f"{(m['no_bet']/total)*100:.1f}%")

    st.subheader("Export Logged Calculations")
    df_logs = export_logs_df()
    st.dataframe(df_logs, use_container_width=True, height=320)
    csv_bytes = df_logs.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, "fair_odds_logs_snapshot.csv", "text/csv")

    if st.button("Overwrite on-disk log with current in-memory logs"):
        try:
            pd.DataFrame(st.session_state["logs"], columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False, line_terminator="\n")
            st.success("Log file overwritten.")
        except Exception as e:
            st.error(f"Failed to write log file: {e}")

    st.subheader("Import / Merge Logs")
    uploaded = st.file_uploader("Upload a logs CSV to merge into history", type=["csv"])
    merge_col1, merge_col2 = st.columns(2)
    with merge_col1:
        dedup_key = st.selectbox(
            "De-duplication key",
            ["timestamp,game,team_a,team_b,adj_gap", "timestamp", "team_a,team_b,adj_gap"],
            help="Used to drop duplicates when merging"
        )
    with merge_col2:
        overwrite_disk = st.checkbox("Overwrite on-disk log with merged result", value=True,
                                     help="If off, we append only new, non-duplicate rows.")

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            cur_df = export_logs_df()

            def keyify(df, keyspec):
                cols = [c.strip() for c in keyspec.split(",")]
                for c in cols:
                    if c not in df.columns:
                        df[c] = ""
                return df.assign(_key=df[cols].astype(str).agg("|".join, axis=1))

            cur_df = keyify(cur_df, dedup_key)
            new_df = keyify(new_df, dedup_key)

            merged = pd.concat([cur_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["_key"]).drop(columns=["_key"])
            merged = merged.sort_values("timestamp", kind="stable")

            for c in LOG_COLUMNS:
                if c not in merged.columns:
                    merged[c] = ""

            st.session_state["logs"] = merged[LOG_COLUMNS].to_dict(orient="records")
            st.session_state["metrics"] = recompute_metrics_from_logs(st.session_state["logs"])

            if overwrite_disk:
                merged[LOG_COLUMNS] = merged.reindex(columns=LOG_COLUMNS, fill_value="")
                merged.to_csv(LOG_PATH, index=False, line_terminator="\n")
            else:
                on_disk = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame(columns=merged.columns)
                on_disk = keyify(on_disk, dedup_key)
                only_new = merged[~merged["_key"].isin(on_disk["_key"])] if not on_disk.empty else merged
                only_new.drop(columns=["_key"]).to_csv(LOG_PATH, mode="a", index=False, header=not LOG_PATH.exists())

            st.success(f"Merged {len(new_df)} rows; total history is now {len(st.session_state['logs'])} rows.")
        except Exception as e:
            st.error(f"Failed to import CSV: {e}")

    st.markdown("---")
    st.subheader("Migrate log schema (fix mixed old/new columns)")
    if st.button("Migrate logs to canonical schema"):
        try:
            n = migrate_log_schema(LOG_PATH)
            st.success(f"Migrated log to canonical schema. Rows written: {n}")
            df_new = pd.read_csv(LOG_PATH)
            for c in LOG_COLUMNS:
                if c not in df_new.columns:
                    df_new[c] = ""
            st.session_state["logs"] = df_new[LOG_COLUMNS].to_dict(orient="records")
            st.session_state["metrics"] = recompute_metrics_from_logs(st.session_state["logs"])
        except Exception as e:
            st.error(f"Migration failed: {e}")

    st.subheader("Repair log file (skip malformed rows and rewrite)")
    if st.button("Repair (re-write clean CSV from parsable rows)"):
        try:
            df_good = read_csv_tolerant(LOG_PATH)
            if df_good.empty:
                st.warning("No rows could be recovered to write.")
            else:
                for c in LOG_COLUMNS:
                    if c not in df_good.columns:
                        df_good[c] = ""
                df_good = df_good.reindex(columns=LOG_COLUMNS, fill_value="")
                df_good.to_csv(LOG_PATH, index=False, line_terminator="\n")
                st.success(f"Repaired and rewrote {LOG_PATH.name} with {len(df_good)} rows.")
                st.session_state["logs"] = df_good[LOG_COLUMNS].to_dict(orient="records")
                st.session_state["metrics"] = recompute_metrics_from_logs(st.session_state["logs"])
        except Exception as e:
            st.error(f"Repair failed: {e}")

    st.caption(f"Persistent log file: {LOG_PATH}")



# --------------------------
# CS2 IN-PLAY INDICATOR (MVP)
# --------------------------
def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))

def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))

def estimate_inplay_prob(p0: float,
                         rounds_a: int,
                         rounds_b: int,
                         econ_a: float = 0.0,
                         econ_b: float = 0.0,
                         pistol_a: Optional[bool] = None,
                         pistol_b: Optional[bool] = None,
                         beta_score: float = 0.22,
                         beta_econ: float = 0.06,
                         beta_pistol: float = 0.35) -> float:
    """
    Simple in-play updater:
    - Starts at pre-match p0 (your fair odds model).
    - Nudges log-odds by score diff + econ diff + optional pistol info.
    This is *not* a finished model — it's an MVP skeleton you can later fit from data.
    """
    score_diff = int(rounds_a) - int(rounds_b)
    econ_diff_k = (float(econ_a) - float(econ_b)) / 1000.0  # assume inputs are $ totals; scale to "per 1k"
    x = _logit(p0) + beta_score * score_diff + beta_econ * econ_diff_k
    if pistol_a is True: x += beta_pistol
    if pistol_b is True: x -= beta_pistol
    return _sigmoid(x)

def estimate_sigma(p0: float,
                   rounds_played: int,
                   econ_a: float = 0.0,
                   econ_b: float = 0.0,
                   chaos_boost: float = 0.0,
                   base: float = 0.18,
                   min_sigma: float = 0.03,
                   max_sigma: float = 0.30) -> float:
    """
    'Certainty band' width (probability-space).
    Wider early; narrows as evidence accrues. Strong pre-match favorite narrows slightly.
    Adds widening if economy is unstable or you manually flag chaos.
    """
    rp = max(int(rounds_played), 0)
    # pregame confidence 0..1
    conf0 = min(1.0, abs(p0 - 0.5) * 2.0)
    # early-game uncertainty term ~ 1/sqrt(rp+1)
    term_time = 1.0 / np.sqrt(rp + 1.0)
    # econ instability (if both low, instability higher) - super rough proxy
    econ_total = max(1.0, float(econ_a) + float(econ_b))
    econ_imbalance = abs(float(econ_a) - float(econ_b)) / econ_total  # 0..1
    econ_term = 0.06 * (1.0 - econ_imbalance)  # if balanced econ (often 'knife edge'), widen a bit
    sigma = base * term_time * (1.0 - 0.35 * conf0) + econ_term + float(chaos_boost)
    return float(min(max_sigma, max(min_sigma, sigma)))

with tabs[3]:
    st.header("CS2 In-Play Indicator (MVP)")
    st.caption("This adds a live 'fair probability' line + certainty bands. MVP = manual inputs (score + econ + optional pistol) plus manual market price.")

    # reuse CS2 teams list for convenience
    df_cs2_live = load_cs2_teams()

    colA, colB = st.columns(2)
    with colA:
        team_a_live = st.selectbox("Team A", df_cs2_live["team"].tolist(), key="live_a")
    with colB:
        team_b_live = st.selectbox("Team B", df_cs2_live["team"].tolist(), key="live_b")

    st.markdown("### Step 1 — Pre-match fair probability (from your model)")
    st.caption("If you've already scraped/calculated in the CS2 tab, you can paste the p_model here. Next step is wiring it to reuse session_state automatically.")
    p0 = st.number_input("Pre-match fair win% for Team A (0–1)", min_value=0.01, max_value=0.99,
                         value=float(st.session_state.get("live_p0", 0.60)),
                         step=0.01, format="%.2f")
    st.session_state["live_p0"] = float(p0)

    st.markdown("### Step 2 — Live inputs (per round or whenever you want to update)")
    if "live_rows" not in st.session_state:
        st.session_state["live_rows"] = []

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rounds_a = st.number_input("Rounds A", 0, 16, int(st.session_state.get("live_rounds_a", 0)), 1)
    with c2:
        rounds_b = st.number_input("Rounds B", 0, 16, int(st.session_state.get("live_rounds_b", 0)), 1)
    with c3:
        econ_a = st.number_input("Team A total economy ($)", min_value=0, value=int(st.session_state.get("live_econ_a", 0)), step=500)
    with c4:
        econ_b = st.number_input("Team B total economy ($)", min_value=0, value=int(st.session_state.get("live_econ_b", 0)), step=500)

    colP1, colP2, colChaos = st.columns(3)
    with colP1:
        pistol_a = st.checkbox("A won most recent pistol?", value=False, key="live_pistol_a")
    with colP2:
        pistol_b = st.checkbox("B won most recent pistol?", value=False, key="live_pistol_b")
    with colChaos:
        chaos_boost = st.slider("Chaos widen (manual)", 0.00, 0.25, float(st.session_state.get("live_chaos", 0.00)), 0.01)

    st.markdown("### Step 3 — Market price input")
    st.caption("Enter market probability/price for Team A (e.g., 0.43 if YES share is $0.43).")
    market_p = st.number_input("Market price/prob for Team A (0–1)", min_value=0.01, max_value=0.99,
                              value=float(st.session_state.get("live_market_p", 0.50)),
                              step=0.01, format="%.2f")
    st.session_state["live_market_p"] = float(market_p)

    st.markdown("### Model knobs (MVP)")
    colB1, colB2, colB3 = st.columns(3)
    with colB1:
        beta_score = st.slider("β score", 0.05, 0.60, float(st.session_state.get("beta_score", 0.22)), 0.01)
    with colB2:
        beta_econ = st.slider("β econ", 0.00, 0.20, float(st.session_state.get("beta_econ", 0.06)), 0.01)
    with colB3:
        beta_pistol = st.slider("β pistol", 0.00, 0.80, float(st.session_state.get("beta_pistol", 0.35)), 0.01)
    st.session_state["beta_score"] = float(beta_score)
    st.session_state["beta_econ"] = float(beta_econ)
    st.session_state["beta_pistol"] = float(beta_pistol)

    rounds_played = int(rounds_a) + int(rounds_b)
    p_hat = estimate_inplay_prob(p0, int(rounds_a), int(rounds_b), float(econ_a), float(econ_b),
                                 pistol_a=bool(pistol_a), pistol_b=bool(pistol_b),
                                 beta_score=float(beta_score), beta_econ=float(beta_econ), beta_pistol=float(beta_pistol))
    sigma = estimate_sigma(p0, rounds_played, float(econ_a), float(econ_b), float(chaos_boost))
    lo = max(0.01, p_hat - 2.0 * sigma)
    hi = min(0.99, p_hat + 2.0 * sigma)

    st.metric("Model fair p(A) now", f"{p_hat*100:.1f}%")
    st.metric("Certainty band (±2σ)", f"[{lo*100:.1f}%, {hi*100:.1f}%]")
    st.metric("Market p(A)", f"{float(market_p)*100:.1f}%")
    st.metric("Deviation (market - fair)", f"{(float(market_p)-p_hat)*100:+.1f} pp")

    colAdd, colClear, colExport = st.columns(3)
    with colAdd:
        if st.button("Add snapshot"):
            st.session_state["live_rows"].append({
                "t": len(st.session_state["live_rows"]),
                "rounds_a": int(rounds_a),
                "rounds_b": int(rounds_b),
                "econ_a": float(econ_a),
                "econ_b": float(econ_b),
                "p0": float(p0),
                "p_hat": float(p_hat),
                "band_lo": float(lo),
                "band_hi": float(hi),
                "market_p": float(market_p),
                "dev_pp": float((float(market_p)-p_hat)*100.0),
            })
    with colClear:
        if st.button("Clear snapshots"):
            st.session_state["live_rows"] = []
    with colExport:
        if st.button("Export snapshots to Diagnostics"):
            # piggyback on existing export tab (shows a dataframe)
            st.session_state["live_export_df"] = pd.DataFrame(st.session_state["live_rows"])

    st.markdown("### Chart")
    if len(st.session_state["live_rows"]) > 0:
        chart_df = pd.DataFrame(st.session_state["live_rows"])
        chart_df = chart_df[["t","p_hat","band_lo","band_hi","market_p"]].set_index("t")
        st.line_chart(chart_df)
        st.dataframe(pd.DataFrame(st.session_state["live_rows"]))
    else:
        st.info("Add at least one snapshot to see the chart.")

