"""Path constants and schema definitions."""

from pathlib import Path

# Project root: parent of this package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT

LOG_PATH = PROJECT_ROOT / "logs" / "fair_odds_logs.csv"

LOG_COLUMNS = [
    "timestamp",
    "game",
    "series_format",
    "market_type",
    "team_a",
    "team_b",
    "tier_a",
    "tier_b",
    "K",
    "decay",
    "floor",
    "raw_a",
    "raw_b",
    "adj_a",
    "adj_b",
    "adj_gap",
    "use_logistic",
    "a",
    "b",
    "use_clip",
    "clip_limit",
    "p_model",
    "odds_a",
    "odds_b",
    "fair_a",
    "fair_b",
    "ev_a_pct",
    "ev_b_pct",
    "p_decide",
    "ev_a_dec",
    "ev_b_dec",
    "decision",
    "decision_reason",
    "min_edge_pct",
    "prob_gap_pp",
    "shrink_target",
    "p_map_model",
    "p_map_decide",
    "draw_k",
    "p_a20",
    "p_draw",
    "p_b02",
    "odds_a20",
    "odds_draw",
    "odds_b02",
    "ev_a20_pct",
    "ev_draw_pct",
    "ev_b02_pct",
    "selected_outcome",
    "selected_prob",
    "selected_odds",
    "selected_ev_pct",
    "fair_a20_us",
    "fair_draw_us",
    "fair_b02_us",
]

INPLAY_LOG_PATH = PROJECT_ROOT / "logs" / "inplay_kappa_logs_clean.csv"
INPLAY_RESULTS_PATH = PROJECT_ROOT / "logs" / "inplay_match_results_clean.csv"
INPLAY_MAP_RESULTS_PATH = PROJECT_ROOT / "logs" / "inplay_map_results_clean.csv"

INPLAY_LOG_COLUMNS = [
    "timestamp",
    "game",
    "match_id",
    "contract_scope",
    "series_format",
    "maps_a_won",
    "maps_b_won",
    "team_a",
    "team_b",
    "map_name",
    "a_side_now",
    "rounds_a",
    "rounds_b",
    "map_index",
    "total_rounds",
    "prev_total_rounds",
    "gap_rounds",
    "gap_flag",
    "gap_reason",
    "buy_state_a",
    "buy_state_b",
    "econ_missing",
    "econ_fragile",
    "pistol_a",
    "pistol_b",
    "gap_delta",
    "streak_winner",
    "streak_len",
    "reversal",
    "n_tracked",
    "p0_map",
    "p_fair_map",
    "kappa_map",
    "p_fair",
    "band_level",
    "band_lo",
    "band_hi",
    "bid",
    "ask",
    "mid",
    "spread_abs",
    "spread_rel",
    "notes",
    "snapshot_idx",
    "half",
    "round_in_half",
    "is_ot",
    "round_in_map",
]

INPLAY_RESULTS_COLUMNS = ["match_id", "game", "team_a", "team_b", "winner", "timestamp"]
INPLAY_MAP_RESULTS_COLUMNS = [
    "match_id",
    "game",
    "map_index",
    "map_name",
    "team_a",
    "team_b",
    "winner",
    "timestamp",
]

KAPPA_CALIB_PATH = PROJECT_ROOT / "config" / "kappa_calibration.json"
KAPPA_TRAIN_SCRIPT = PROJECT_ROOT / "ML" / "train_kappa_calibration.py"
P_CALIB_PATH = PROJECT_ROOT / "config" / "p_calibration.json"
P_CALIB_REPORT_PATH = PROJECT_ROOT / "config" / "p_calibration_report.json"
P_TRAIN_SCRIPT = PROJECT_ROOT / "ML" / "train_prob_calibration.py"
