"""Calibration loading and trainer runners."""
import json
import subprocess
import sys
from pathlib import Path

from .paths import (
    PROJECT_ROOT,
    KAPPA_CALIB_PATH,
    KAPPA_TRAIN_SCRIPT,
    P_CALIB_PATH,
    P_CALIB_REPORT_PATH,
    P_TRAIN_SCRIPT,
)


def load_kappa_calibration() -> dict:
    try:
        if KAPPA_CALIB_PATH.exists():
            return json.loads(KAPPA_CALIB_PATH.read_text())
    except Exception:
        pass
    return {}


def load_p_calibration_json(project_dir: Path) -> dict:
    """Load p_calibration.json if present. Returns {} if missing/invalid."""
    try:
        p = project_dir / "config" / "p_calibration.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def apply_p_calibration(p_raw: float, p_calib: dict, game_key: str) -> float:
    """Apply probability calibration using knots stored in p_calibration.json.

    Robust to BOTH knot formats:
      1) [{"x": 0.2, "y": 0.25}, ...]
      2) [[0.2, 0.25], ...]   (current trainer output)

    Returns p_raw unchanged if calibration is missing/invalid.
    """
    try:
        import bisect

        # Parse/clamp input
        try:
            p = float(p_raw)
        except Exception:
            return float(p_raw)
        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0

        if not isinstance(p_calib, dict):
            return p

        gkey = (game_key or "").strip()
        g = p_calib.get(gkey.lower(), p_calib.get(gkey.upper(), p_calib.get(gkey, {})))
        if not isinstance(g, dict):
            return p

        knots = g.get("knots")
        if not isinstance(knots, list) or len(knots) < 2:
            return p

        xs: list[float] = []
        ys: list[float] = []

        for k in knots:
            x = y = None

            # Format 1: dict
            if isinstance(k, dict):
                x = k.get("x")
                y = k.get("y")
            # Format 2: list/tuple pair
            elif isinstance(k, (list, tuple)) and len(k) >= 2:
                x, y = k[0], k[1]

            try:
                x = float(x)
                y = float(y)
            except Exception:
                continue

            # Skip nonsense
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                continue

            xs.append(x)
            ys.append(y)

        if len(xs) < 2:
            return p

        # Sort by x and de-duplicate identical x values (keep last y)
        pairs = sorted(zip(xs, ys), key=lambda t: t[0])
        xs_sorted: list[float] = []
        ys_sorted: list[float] = []
        for x, y in pairs:
            if xs_sorted and x == xs_sorted[-1]:
                ys_sorted[-1] = y
            else:
                xs_sorted.append(x)
                ys_sorted.append(y)

        if len(xs_sorted) < 2:
            return p

        # Piecewise-linear interpolation with boundary extension
        if p <= xs_sorted[0]:
            return ys_sorted[0]
        if p >= xs_sorted[-1]:
            return ys_sorted[-1]

        i = bisect.bisect_right(xs_sorted, p) - 1
        x0, y0 = xs_sorted[i], ys_sorted[i]
        x1, y1 = xs_sorted[i + 1], ys_sorted[i + 1]
        if x1 == x0:
            return y0
        t = (p - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)
    except Exception:
        return float(p_raw)


def _phase_bucket(total_rounds: int, is_ot: bool) -> str:
    if is_ot:
        return "ot"
    if total_rounds <= 8:
        return "early"
    if total_rounds <= 18:
        return "mid"
    return "late"


def get_kappa_multiplier(calib: dict, game_key: str, band_level: float, total_rounds: int, is_ot: bool) -> float:
    if not calib:
        return 1.0
    g = calib.get(game_key.lower(), calib.get(game_key.upper(), {}))
    if not isinstance(g, dict):
        return 1.0
    band_key = str(band_level)
    if band_key not in g:
        band_key = f"{band_level:.2f}"
    band_obj = g.get(band_key, g.get(str(round(band_level, 2)), {}))
    if not isinstance(band_obj, dict):
        return 1.0
    bucket = _phase_bucket(int(total_rounds), bool(is_ot))
    mult = band_obj.get(bucket, 1.0)
    try:
        return float(mult)
    except Exception:
        return 1.0


def run_kappa_trainer() -> tuple[bool, str]:
    if not KAPPA_TRAIN_SCRIPT.exists():
        return False, f"Trainer not found: {KAPPA_TRAIN_SCRIPT}"
    try:
        proc = subprocess.run(
            [sys.executable, str(KAPPA_TRAIN_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if proc.returncode != 0:
            return False, out.strip() or f"Trainer failed (code {proc.returncode})."
        return True, out.strip() or "Trainer completed."
    except Exception as e:
        return False, f"Trainer error: {e}"


def load_p_calibration() -> dict:
    try:
        if P_CALIB_PATH.exists():
            return json.loads(P_CALIB_PATH.read_text())
    except Exception:
        pass
    return {}


def run_prob_trainer() -> tuple[bool, str]:
    if not P_TRAIN_SCRIPT.exists():
        return False, f"Trainer not found: {P_TRAIN_SCRIPT}"
    try:
        proc = subprocess.run(
            [sys.executable, str(P_TRAIN_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if proc.returncode != 0:
            return False, out.strip() or f"Trainer failed (code {proc.returncode})."
        return True, out.strip() or "Trainer completed."
    except Exception as e:
        return False, f"Trainer error: {e}"


def apply_p_calibration_isotonic(p: float, calib: dict, game_key: str) -> float:
    """Apply piecewise-linear isotonic knots to a probability."""
    try:
        g = calib.get(game_key.lower(), calib.get(game_key.upper(), {}))
        knots = g.get("knots", [])
        if not knots:
            return float(p)
        xs = [float(k[0]) for k in knots]
        ys = [float(k[1]) for k in knots]
        xs2 = [0.0] + xs + [1.0]
        ys2 = [ys[0]] + ys + [ys[-1]]
        x = min(max(float(p), 0.0), 1.0)
        for j in range(len(xs2) - 1):
            if x <= xs2[j + 1]:
                x0, x1 = xs2[j], xs2[j + 1]
                y0, y1 = ys2[j], ys2[j + 1]
                if x1 == x0:
                    return float(y1)
                t = (x - x0) / (x1 - x0)
                return float(y0 + t * (y1 - y0))
        return float(ys2[-1])
    except Exception:
        return float(p)
