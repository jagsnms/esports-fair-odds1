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
LOG_COLUMNS = [
    "timestamp","game","team_a","team_b","tier_a","tier_b",
    "K","decay","floor",
    "raw_a","raw_b","adj_a","adj_b","adj_gap",
    "use_logistic","a","b","use_clip","clip_limit",
    "p_model","odds_a","odds_b","fair_a","fair_b","ev_a_pct","ev_b_pct",
    # Decision-layer fields (newer schema)
    "p_decide","ev_a_dec","ev_b_dec","decision","decision_reason",
    "min_edge_pct","prob_gap_pp","shrink_target",
    # Optional future field (if you add realized outcomes)
    # "outcome"
]

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
    required = ["team", "tier", "rank", "slug", "opendota_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Dota file missing columns: {missing}")
        return df.iloc[0:0]
    df["slug"] = df["slug"].fillna(df["team"].astype(str).str.strip().str.replace(" ", "_"))
    df["tier"] = _coerce_numeric(df["tier"], "Dota tier")
    df["opendota_id"] = _coerce_numeric(df["opendota_id"], "Dota opendota_id")
    bad = df["opendota_id"].isna() | df["tier"].isna() | (df["team"].astype(str).str.strip() == "")
    if bad.any():
        st.warning(f"Dota: dropping {int(bad.sum())} row(s) with missing team/tier/opendota_id.")
        st.dataframe(df.loc[bad, required], use_container_width=True, height=140)
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
                    best_score = score
                    best_idx = idx
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
    path = APP_DIR / "dota2_gosu_rankings.csv"
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
            out.append({"opponent": opp, "win": win})
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
            matches.append({"opponent": opp_name, "win": win})
        return matches
    except Exception as e:
        st.error(f"Error fetching OpenDota data: {e}")
        return []

# ==========================
# Scoring Logic with Tier-Based + Recency Adjustments
# ==========================
def calculate_score(matches,
                    df: pd.DataFrame,
                    current_opponent_tier=None,
                    weight_scheme: str = "piecewise",
                    K: int = 6,
                    decay: float = 0.85,
                    floor: float = 0.6,
                    newest_first: bool = True):
    """
    weight_scheme:
      - "piecewise": last K at full weight, older decay to 'floor', sum-preserving
      - anything else -> flat weights (no recency)
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
        win = match["win"]
        tier = get_team_tier(opp, df)

        # Base tier points (unchanged)
        if win:
            if tier == 1: points = 4
            elif tier == 1.5: points = 3
            elif tier == 2: points = 2.5
            elif tier == 3: points = 2
            elif tier == 4: points = 1.5
            else: points = 1
        else:
            if tier == 1: points = -1
            elif tier == 1.5: points = -1.5
            elif tier == 2: points = -2
            elif tier == 3: points = -2.5
            elif tier == 4: points = -3
            else: points = -4

        raw_score += points  # audit

        # ---- Tier-gap weighting (COMPRESSED to ~0.85–1.15) ----
        if current_opponent_tier is not None:
            tier_gap = tier - current_opponent_tier
            if win:
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

        # Compress toward 1 so the final band is ~0.85–1.15
        weight_tier = 1 + (weight_tier_old - 1) * 0.3
        weight_tier = max(0.85, min(weight_tier, 1.15))

        # Apply recency weight on top of tier-gap weight
        w_match = weights[i]
        adj_points = points * weight_tier * w_match
        adjusted_score += adj_points

        breakdown.append(
            f"{'Win' if win else 'Loss'} vs {opp} (OppTier {tier}, Gap {tier_gap:.1f}) "
            f"Pts={points:+.2f} × TierW={weight_tier:.2f} × RecW={w_match:.3f} = {adj_points:+.3f}"
        )

    return raw_score, adjusted_score, breakdown

# ==========================
# Odds helpers + mapping
# ==========================
def american_to_decimal(odds: int) -> float:
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds)) + 1

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
                # skip malformed lines
                continue
            # normalize to canonical schema
            rows.append({k: d.get(k, "") for k in LOG_COLUMNS})

    # de-dupe loose key
    seen = set()
    deduped = []
    for d in rows:
        key = (d.get("timestamp",""), d.get("game",""), d.get("team_a",""), d.get("team_b",""), d.get("adj_gap",""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)

    # sort by timestamp (best effort)
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
    """Rebuild Diagnostics counters from logged entries (uses decision EVs if present)."""
    metrics = {"total": 0, "fav_value": 0, "dog_value": 0, "no_bet": 0}
    for r in rows:
        metrics["total"] += 1
        try:
            odds_a = int(r["odds_a"]); odds_b = int(r["odds_b"])
            ev_a = float(r.get("ev_a_dec", r.get("ev_a_pct", -1e9)))
            ev_b = float(r.get("ev_b_dec", r.get("ev_b_pct", -1e9)))
        except Exception:
            metrics["no_bet"] += 1
            continue
        imp_a = implied_prob_from_american(odds_a)
        imp_b = implied_prob_from_american(odds_b)
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
        # Recover by skipping malformed lines so the app can boot
        df = read_csv_tolerant(LOG_PATH)
        st.warning(f"Loaded log after skipping malformed rows. Rows kept: {len(df)}")
    except Exception as e:
        st.error(f"Error reading log CSV: {e}")
        return []
    return df.to_dict(orient="records")

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

def update_metrics(ev_a: float, ev_b: float, odds_a: int, odds_b: int):
    # Quick counter; full recompute runs after we persist
    m = st.session_state.get("metrics", {"total": 0, "fav_value": 0, "dog_value": 0, "no_bet": 0})
    m["total"] += 1
    imp_a = implied_prob_from_american(odds_a)
    imp_b = implied_prob_from_american(odds_b)
    market_fav = "A" if imp_a >= imp_b else "B"
    if ev_a > 0 or ev_b > 0:
        pick = "A" if ev_a >= ev_b else "B"
        if (ev_a if pick == "A" else ev_b) <= 0:
            m["no_bet"] += 1
        else:
            if pick == market_fav: m["fav_value"] += 1
            else: m["dog_value"] += 1
    else:
        m["no_bet"] += 1
    st.session_state["metrics"] = m

def log_row(entry: dict):
    # Append to memory + persist + recompute metrics from full history
    st.session_state["logs"].append({k: entry.get(k, "") for k in LOG_COLUMNS})
    try:
        persist_log_row(entry)
    except Exception as e:
        st.warning(f"Could not persist log row: {e}")
    st.session_state["metrics"] = recompute_metrics_from_logs(st.session_state["logs"])

def export_logs_df() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.get("logs", []), columns=LOG_COLUMNS)

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

tabs = st.tabs(["CS2", "Dota2", "Diagnostics / Export"])

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
        hltv_a_val = row_a["hltv_id"]  # may be NaN
        team_a_id = str(int(hltv_a_val)) if pd.notna(hltv_a_val) else None

        # Label to indicate scrapability
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
        hltv_b_val = row_b["hltv_id"]  # may be NaN
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

    odds_a = st.number_input("Team A Market Odds (CS2)", value=-140, key="cs2_odds_a")
    odds_b = st.number_input("Team B Market Odds (CS2)", value=+120, key="cs2_odds_b")

    # Recency controls
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
                weight_scheme="piecewise", K=K_cs2, decay=decay_cs2, floor=floor_cs2, newest_first=True
            )
            raw_b, adj_b, breakdown_b = calculate_score(
                matches_b, df_cs2, current_opponent_tier=team_a_tier,
                weight_scheme="piecewise", K=K_cs2, decay=decay_cs2, floor=floor_cs2, newest_first=True
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

            # ----- Decision layer (classification + Summary line) -----
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
                ev_a_eff = ev_b_eff = -1e9
            else:
                pick_team = team_a if decision["choice"] == "A" else team_b
                pick_ev = decision["ev_a_dec"] if decision["choice"] == "A" else decision["ev_b_dec"]
                st.markdown(f"**Decision:** ✅ Bet **{pick_team}** ({pick_ev:+.2f}% EV) — {decision['reason']}")
                ev_a_eff = decision["ev_a_dec"] if decision["choice"] == "A" else -1e9
                ev_b_eff = decision["ev_b_dec"] if decision["choice"] == "B" else -1e9

            # Diagnostics & logging use decision EVs
            update_metrics(ev_a_eff, ev_b_eff, odds_a, odds_b)
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "game": "CS2",
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
            }
            log_row(entry)
        else:
            st.warning("Scrape both teams first.")

# --------------------------
# Dota TAB
# --------------------------
with tabs[1]:
    st.header("Dota 2 Fair Odds")
    df_dota = load_dota_teams()
    dota_ok = df_dota.dropna(subset=["opendota_id"])

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

    # Recency controls
    st.subheader("Recency Weighting (Dota2)")
    K_dota = st.slider("Full-weight recent matches (K)", 3, 12, 6, key="K_dota")
    decay_dota = st.slider("Decay per step beyond K", 0.75, 0.95, 0.85, 0.01, key="decay_dota")
    floor_dota = st.slider("Minimum weight floor", 0.40, 0.90, 0.60, 0.01, key="floor_dota")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A (Dota2)", dota_ok["team"].tolist(), key="dota_a")
        row_a = dota_ok.loc[dota_ok["team"] == team_a].iloc[0]
        team_a_id = int(row_a["opendota_id"])
        team_a_slug = row_a["slug"]
        if st.button("Scrape Team A (Dota2)"):
            if src.startswith("OpenDota"):
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
        team_b_id = int(row_b["opendota_id"])
        team_b_slug = row_b["slug"]
        if st.button("Scrape Team B (Dota2)"):
            if src.startswith("OpenDota"):
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

    odds_a = st.number_input("Team A Market Odds (Dota2)", value=-140, key="dota_odds_a")
    odds_b = st.number_input("Team B Market Odds (Dota2)", value=+120, key="dota_odds_b")

    if st.button("Calculate (Dota2)"):
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
                    weight_scheme="piecewise", K=K_dota, decay=decay_dota, floor=floor_dota, newest_first=True
                )
                raw_b, adj_b, breakdown_b = calculate_score(
                    matches_b, df_dota, current_opponent_tier=team_a_tier,
                    weight_scheme="piecewise", K=K_dota, decay=decay_dota, floor=floor_dota, newest_first=True
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

                st.subheader("Summary (Dota2)")
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
                    ev_a_eff = ev_b_eff = -1e9
                else:
                    pick_team = team_a if decision["choice"] == "A" else team_b
                    pick_ev = decision["ev_a_dec"] if decision["choice"] == "A" else decision["ev_b_dec"]
                    st.markdown(f"**Decision:** ✅ Bet **{pick_team}** ({pick_ev:+.2f}% EV) — {decision['reason']}")
                    ev_a_eff = decision["ev_a_dec"] if decision["choice"] == "A" else -1e9
                    ev_b_eff = decision["ev_b_dec"] if decision["choice"] == "B" else -1e9

                update_metrics(ev_a_eff, ev_b_eff, odds_a, odds_b)
                entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "game": "Dota2",
                    "team_a": team_a, "team_b": team_b,
                    "tier_a": team_a_tier, "tier_b": team_b_tier,
                    "K": K_dota, "decay": decay_dota, "floor": floor_dota,
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

    # Overwrite from memory
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

            st.session_state["logs"] = merged.to_dict(orient="records")
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
            st.session_state["logs"] = df_new.to_dict(orient="records")
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
                # Ensure canonical header order
                df_good = df_good.reindex(columns=LOG_COLUMNS, fill_value="")
                df_good.to_csv(LOG_PATH, index=False, line_terminator="\n")
                st.success(f"Repaired and rewrote {LOG_PATH.name} with {len(df_good)} rows.")
                st.session_state["logs"] = df_good.to_dict(orient="records")
                st.session_state["metrics"] = recompute_metrics_from_logs(st.session_state["logs"])
        except Exception as e:
            st.error(f"Repair failed: {e}")

    st.caption(f"Persistent log file: {LOG_PATH}")
