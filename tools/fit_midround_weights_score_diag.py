"""
Fit midround term weights from score_diag_v2 using term_raw (learn "true" coefficients).

Uses logs/history_score_points.jsonl (schema score_diag_v2) + labels from
logs/history_points.jsonl to fit a logistic regression estimating the marginal effect
of each raw term on round-win probability.

Outputs: out/midround_fit_weights.json, out/midround_fit_weights.csv, out/midround_fit_calibration_bins.csv

Usage (repo root):
  python tools/fit_midround_weights_score_diag.py
  python tools/fit_midround_weights_score_diag.py --C 0.5 --test_split 0.25 --features alive,hp,loadout,bomb
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_IGNORE_REASONS = "no_source,no_compute,inter_map_break,replay_loop,passthrough"
DEFAULT_FEATURES = "alive,hp,loadout,bomb"


def _safe_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _parse_ignore_reasons(s: str) -> set[str]:
    if not s.strip():
        return set()
    return {r.strip() for r in s.split(",") if r.strip()}


def _ridge_logistic_fit_scipy(
    X: np.ndarray, y: np.ndarray, C: float, seed: int
) -> tuple[np.ndarray, float]:
    """Ridge logistic regression via scipy. X is (n_samples, n_features), y in {0,1}. Returns (coef, intercept)."""
    from scipy.optimize import minimize
    from scipy.special import expit, log1p

    n_samples, n_features = X.shape
    rng = np.random.default_rng(seed)
    w0 = rng.standard_normal(n_features) * 0.01
    b0 = 0.0
    x0 = np.concatenate([w0, [b0]])

    def nll(x: np.ndarray) -> float:
        w = x[:-1]
        b = x[-1]
        logits = X @ w + b
        logits = np.clip(logits, -20.0, 20.0)
        p = expit(logits)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        nll_val = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / n_samples
        reg = (1.0 / (2.0 * C)) * np.sum(w * w)
        return nll_val + reg

    def grad(x: np.ndarray) -> np.ndarray:
        w = x[:-1]
        b = x[-1]
        logits = X @ w + b
        logits = np.clip(logits, -20.0, 20.0)
        p = expit(logits)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        err = p - y
        gw = (X.T @ err) / n_samples + (1.0 / C) * w
        gb = np.sum(err) / n_samples
        return np.concatenate([gw, [gb]])

    res = minimize(nll, x0, method="L-BFGS-B", jac=grad)
    w = res.x[:-1]
    b = float(res.x[-1])
    return w, b


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit midround term weights from score_diag_v2 (term_raw) with logistic regression"
    )
    parser.add_argument(
        "--score_input",
        default="logs/history_score_points.jsonl",
        help="Score diagnostics JSONL (score_diag_v2)",
    )
    parser.add_argument(
        "--label_input",
        default="logs/history_points.jsonl",
        help="Label source JSONL (round_result events)",
    )
    parser.add_argument(
        "--phase",
        default="IN_PROGRESS",
        help="Phase filter (default IN_PROGRESS)",
    )
    parser.add_argument(
        "--ignore_reasons",
        default=DEFAULT_IGNORE_REASONS,
        help="Comma-separated clamp_reason values to exclude",
    )
    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES,
        help="Comma-separated feature names (default alive,hp,loadout,bomb)",
    )
    parser.add_argument("--C", type=float, default=1.0, help="L2 inverse strength (default 1.0)")
    parser.add_argument("--test_split", type=float, default=0.2, help="Test fraction (stratified, default 0.2)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed (default 1337)")
    parser.add_argument("--out_dir", default="out", help="Output directory")
    parser.add_argument("--audit_out", default="midround_fit_audit_rows.csv", help="Audit CSV path (relative to out_dir) or absolute; empty to skip")
    parser.add_argument("--audit_n", type=int, default=300, help="Max number of rows to write to audit CSV (default 300)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    # Allow absolute paths for inputs so tests can pass temp dirs
    def _resolve_input_path(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else repo_root / path
    score_path = _resolve_input_path(args.score_input)
    label_path = _resolve_input_path(args.label_input)
    out_path = repo_root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    feature_names = [f.strip() for f in args.features.split(",") if f.strip()]
    if not feature_names:
        feature_names = list(DEFAULT_FEATURES.split(","))
    ignore_reasons = _parse_ignore_reasons(args.ignore_reasons)
    phase_filter = args.phase.strip() if args.phase and args.phase.strip() else None

    # 1) Build labels from label_input
    label_lines = _read_jsonl(label_path)
    labels: dict[tuple[int, int, int], int] = {}
    for obj in label_lines:
        if not isinstance(obj, dict):
            continue
        ev = obj.get("event") if isinstance(obj.get("event"), dict) else None
        if not ev or ev.get("event_type") != "round_result":
            continue
        gn = ev.get("game_number")
        mi = ev.get("map_index") or obj.get("map_index")
        rn = ev.get("round_number")
        if rn is None or mi is None:
            continue
        try:
            gn = int(gn) if gn is not None else 0
            mi = int(mi)
            rn = int(rn)
        except (TypeError, ValueError):
            continue
        winner_a = ev.get("round_winner_is_team_a")
        if winner_a is not None:
            labels[(gn, mi, rn)] = 1 if winner_a else 0

    # 2) Read score_diag_v2, filter, extract term_raw and term_coef; build audit row metadata per accepted row
    score_lines = _read_jsonl(score_path)
    rows: list[tuple[tuple[int, int, int], list[float], dict[str, float] | None]] = []
    audit_rows: list[dict[str, Any]] = []
    term_coefs_per_feature: dict[str, list[float]] = {f: [] for f in feature_names}

    for obj in score_lines:
        if obj.get("schema") != "score_diag_v2":
            continue
        phase = obj.get("phase")
        if phase == "idle":
            continue
        if phase_filter is not None and phase != phase_filter:
            continue
        clamp_reason = obj.get("clamp_reason")
        if clamp_reason is not None and clamp_reason in ignore_reasons:
            continue
        gn = obj.get("game_number")
        mi = obj.get("map_index")
        rn = obj.get("round_number")
        if mi is None or rn is None:
            continue
        try:
            gn = int(gn) if gn is not None else 0
            mi = int(mi)
            rn = int(rn)
        except (TypeError, ValueError):
            continue
        key = (gn, mi, rn)
        if key not in labels:
            continue
        term_raw = obj.get("term_raw")
        term_coef = obj.get("term_coef")
        if not isinstance(term_raw, dict):
            continue
        y = labels[key]
        feat_vec: list[float] = []
        missing = False
        for f in feature_names:
            v = term_raw.get(f)
            if v is None and f in term_raw:
                v = term_raw[f]
            if v is None:
                missing = True
                break
            try:
                feat_vec.append(float(v))
            except (TypeError, ValueError):
                missing = True
                break
        if missing or len(feat_vec) != len(feature_names):
            continue
        rows.append((key, feat_vec, term_coef if isinstance(term_coef, dict) else None))
        if isinstance(term_coef, dict):
            for f in feature_names:
                if f in term_coef and term_coef[f] is not None:
                    try:
                        term_coefs_per_feature[f].append(float(term_coef[f]))
                    except (TypeError, ValueError):
                        pass
        # Audit row: metadata for label-alignment validation
        audit: dict[str, Any] = {
            "game_number": gn,
            "map_index": mi,
            "round_number": rn,
            "y": y,
            "phase": phase,
            "clamp_reason": clamp_reason,
            "score_raw": _safe_float(obj.get("score_raw")),
            "p_unshaped": obj.get("p_unshaped") if obj.get("p_unshaped") is not None else "",
            "p_hat_final": _safe_float(obj.get("p_hat_final")),
        }
        if isinstance(term_raw, dict):
            for k, v in term_raw.items():
                if isinstance(v, (int, float)):
                    audit[f"term_raw_{k}"] = v
        term_contribs = obj.get("term_contribs")
        if isinstance(term_contribs, dict):
            for k, v in term_contribs.items():
                if isinstance(v, (int, float)):
                    audit[f"term_contribs_{k}"] = v
        for k, v in obj.items():
            if k.startswith("team_") or k in ("a_side", "team_a_is_team_one"):
                if v is not None and k not in audit:
                    audit[k] = v
        audit_rows.append(audit)

    if not rows:
        print("No rows after filtering. Ensure score_diag_v2 data and labels exist.")
        return

    X = np.array([r[1] for r in rows], dtype=float)
    y = np.array([labels[r[0]] for r in rows], dtype=float)
    n_rows = len(y)
    class_balance = {"n": int(n_rows), "n_pos": int(np.sum(y)), "n_neg": int(n_rows - np.sum(y)), "rate_pos": float(np.mean(y))}

    # Direction sanity: mean(score_raw) and mean(p_unshaped) by y=0 and y=1 (validate label alignment)
    score_raw_arr = np.array([a.get("score_raw", float("nan")) for a in audit_rows], dtype=float)
    p_unshaped_arr: list[float] = []
    for a in audit_rows:
        v = a.get("p_unshaped")
        if v is None or v == "":
            p_unshaped_arr.append(float("nan"))
        else:
            try:
                p_unshaped_arr.append(float(v))
            except (TypeError, ValueError):
                p_unshaped_arr.append(float("nan"))
    p_unshaped_arr = np.array(p_unshaped_arr, dtype=float)
    mask0 = y == 0
    mask1 = y == 1
    mean_score_raw_y0 = float(np.nanmean(score_raw_arr[mask0])) if np.any(mask0) else float("nan")
    mean_score_raw_y1 = float(np.nanmean(score_raw_arr[mask1])) if np.any(mask1) else float("nan")
    mean_p_unshaped_y0 = float(np.nanmean(p_unshaped_arr[mask0])) if np.any(mask0) else float("nan")
    mean_p_unshaped_y1 = float(np.nanmean(p_unshaped_arr[mask1])) if np.any(mask1) else float("nan")
    mean_by_class = {
        "mean_score_raw_y0": mean_score_raw_y0,
        "mean_score_raw_y1": mean_score_raw_y1,
        "mean_p_unshaped_y0": mean_p_unshaped_y0,
        "mean_p_unshaped_y1": mean_p_unshaped_y1,
    }
    print("Direction sanity (label alignment):")
    print(f"  mean(score_raw)  y=0: {mean_score_raw_y0:.6f}  y=1: {mean_score_raw_y1:.6f}")
    print(f"  mean(p_unshaped) y=0: {mean_p_unshaped_y0:.6f}  y=1: {mean_p_unshaped_y1:.6f}")
    if mean_score_raw_y1 < mean_score_raw_y0 or (not np.isnan(mean_p_unshaped_y1) and not np.isnan(mean_p_unshaped_y0) and mean_p_unshaped_y1 < mean_p_unshaped_y0):
        print("  WARNING: y=1 has LOWER means than y=0 — labels may be inverted or Team A identity misaligned.")
    else:
        print("  OK: y=1 has higher (or equal) score/prob means than y=0.")

    # current_term_coef: median of logged term_coef per feature (or from first row)
    current_term_coef: dict[str, float] = {}
    for f in feature_names:
        vals = term_coefs_per_feature.get(f) or []
        if vals:
            current_term_coef[f] = float(np.median(vals))
        else:
            # fallback from first row with term_coef
            for _, _, tc in rows:
                if tc and f in tc and tc[f] is not None:
                    try:
                        current_term_coef[f] = float(tc[f])
                        break
                    except (TypeError, ValueError):
                        pass
            if f not in current_term_coef:
                current_term_coef[f] = 0.0

    # 3) Stratified train/test split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    n_test = max(1, int(n_rows * args.test_split))
    n_train = n_rows - n_test
    # simple shuffle (stratified would require splitting per class; for simplicity we shuffle and take last n_test)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 4) Standardize (fit on train)
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    stds[stds < 1e-10] = 1.0
    X_train_std = (X_train - means) / stds
    X_test_std = (X_test - means) / stds

    # 5) Fit logistic regression
    use_sklearn = False
    model = None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X_std_all = (X - means) / stds
        try:
            X_train_std, X_test_std, y_train, y_test = train_test_split(
                X_std_all, y, test_size=args.test_split, stratify=y, random_state=args.seed
            )
        except ValueError:
            X_train_std, X_test_std, y_train, y_test = train_test_split(
                X_std_all, y, test_size=args.test_split, random_state=args.seed
            )
        X_train = X_train_std * stds + means  # keep for suggested_coef
        X_test = X_test_std * stds + means
        n_train = len(y_train)
        n_test = len(y_test)
        model = LogisticRegression(C=args.C, penalty="l2", solver="lbfgs", max_iter=1000, random_state=args.seed)
        model.fit(X_train_std, y_train)
        coef_std = model.coef_.ravel().copy()
        intercept_std = float(model.intercept_[0])
        use_sklearn = True
    except ImportError:
        X_train_std = (X_train - means) / stds
        X_test_std = (X_test - means) / stds
        coef_std, intercept_std = _ridge_logistic_fit_scipy(X_train_std, y_train, args.C, args.seed)
        coef_std = np.asarray(coef_std)

    # 6) Coef in raw (unstandardized) space
    coef_unstd = coef_std / stds
    intercept_unstd = intercept_std - np.sum(coef_std * means / stds)

    # 7) P(y=1) for metrics and calibration: use predict_proba with correct class when sklearn
    if use_sklearn and model is not None:
        proba_train = model.predict_proba(X_train_std)
        proba_test = model.predict_proba(X_test_std)
        classes = list(model.classes_)
        if 1 in classes:
            idx_pos = classes.index(1)
        else:
            idx_pos = int(np.argmax(classes))
        y_prob_train = np.clip(proba_train[:, idx_pos].astype(float), 1e-15, 1 - 1e-15)
        y_prob_test = np.clip(proba_test[:, idx_pos].astype(float), 1e-15, 1 - 1e-15)
    else:
        logits_train = X_train_std @ coef_std + intercept_std
        logits_test = X_test_std @ coef_std + intercept_std
        y_prob_train = np.clip(1.0 / (1.0 + np.exp(-np.clip(logits_train, -20, 20))), 1e-15, 1 - 1e-15)
        y_prob_test = np.clip(1.0 / (1.0 + np.exp(-np.clip(logits_test, -20, 20))), 1e-15, 1 - 1e-15)
        # Sanity: P(y=1) should correlate positively with linear score
        score_linear = X_train_std @ coef_std
        corr = np.corrcoef(y_prob_train, score_linear)[0, 1] if len(y_prob_train) > 1 else 1.0
        assert corr > 0, f"scipy path: y_prob and score correlation should be positive, got {corr}"

    def metrics_from_proba(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
        n_pos = int(np.sum(y_true))
        n_neg = len(y_true) - n_pos
        if n_pos > 0 and n_neg > 0:
            order = np.argsort(y_prob)[::-1]
            y_ord = y_true[order]
            rank = np.arange(1, len(y_ord) + 1, dtype=float)
            auc = (np.sum(rank[y_ord == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        else:
            auc = 0.5
        logloss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        brier = np.mean((y_prob - y_true) ** 2)
        return {"auc": float(auc), "logloss": float(logloss), "brier": float(brier)}

    metrics_train = metrics_from_proba(y_train, y_prob_train)
    metrics_test = metrics_from_proba(y_test, y_prob_test)

    # Ensure we report P(y=1): if AUC < 0.5 we are using wrong class column or inverted fit — flip probs and coefs
    if metrics_train["auc"] < 0.5:
        y_prob_train = 1.0 - y_prob_train
        y_prob_test = 1.0 - y_prob_test
        coef_std = -coef_std
        intercept_std = -intercept_std
        coef_unstd = coef_std / stds
        intercept_unstd = intercept_std - np.sum(coef_std * means / stds)
        metrics_train = metrics_from_proba(y_train, y_prob_train)
        metrics_test = metrics_from_proba(y_test, y_prob_test)

    # 8) Calibration bins (10 bins) on test — use y_prob_test (P(y=1))
    p_test = y_prob_test
    n_bins = 10
    bin_edges = np.percentile(p_test, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        bin_edges = np.array([0.0, 1.0])
    cal_rows: list[dict[str, Any]] = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (p_test >= lo) & (p_test < hi) if i < len(bin_edges) - 2 else (p_test >= lo) & (p_test <= hi)
        if np.sum(mask) == 0:
            continue
        mean_pred = float(np.mean(p_test[mask]))
        mean_actual = float(np.mean(y_test[mask]))
        count = int(np.sum(mask))
        cal_rows.append({"bin_lo": lo, "bin_hi": hi, "mean_pred": mean_pred, "mean_actual": mean_actual, "count": count})
    if not cal_rows and len(p_test) > 0:
        cal_rows.append({
            "bin_lo": float(np.min(p_test)),
            "bin_hi": float(np.max(p_test)),
            "mean_pred": float(np.mean(p_test)),
            "mean_actual": float(np.mean(y_test)),
            "count": len(p_test),
        })

    # 9) Suggested coef: scale coef_unstd so RMS(X @ suggested) ≈ RMS(X @ current_coef) on train (linear part only)
    current_coef_vec = np.array([current_term_coef.get(f, 0.0) for f in feature_names])
    score_current = X_train @ current_coef_vec
    score_fitted = X_train @ coef_unstd
    rms_current = np.sqrt(np.mean(score_current ** 2))
    rms_fitted = np.sqrt(np.mean(score_fitted ** 2))
    if rms_fitted > 1e-12:
        g = float(rms_current / rms_fitted)
    else:
        g = 1.0
    suggested_coef = coef_unstd * g

    # 10) Odds ratio per 1 std (on standardized feature): exp(coef_std)
    odds_ratio_per_1std = np.exp(coef_std)

    # Outputs
    weights_json: dict[str, Any] = {
        "n_rows": n_rows,
        "class_balance": class_balance,
        "mean_by_class": mean_by_class,
        "feature_names": feature_names,
        "coef_std": coef_std.tolist(),
        "intercept_std": float(intercept_std),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "coef_unstd": coef_unstd.tolist(),
        "intercept_unstd": float(intercept_unstd),
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "current_term_coef": current_term_coef,
        "suggested_coef": {f: float(suggested_coef[i]) for i, f in enumerate(feature_names)},
        "scale_g": float(g),
    }
    json_path = out_path / "midround_fit_weights.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(weights_json, f, indent=2)
    print(f"Wrote {json_path}")

    csv_path = out_path / "midround_fit_weights.csv"
    with open(csv_path, "w", encoding="utf-8") as csv_file:
        csv_file.write("feature,coef_unstd,coef_std,odds_ratio_per_1std,current_coef,suggested_coef\n")
        for i, feat in enumerate(feature_names):
            csv_file.write(
                f"{feat},{coef_unstd[i]:.8f},{coef_std[i]:.8f},{odds_ratio_per_1std[i]:.6f},"
                f"{current_term_coef.get(feat, 0):.8f},{suggested_coef[i]:.8f}\n"
            )
    print(f"Wrote {csv_path}")

    cal_path = out_path / "midround_fit_calibration_bins.csv"
    with open(cal_path, "w", encoding="utf-8") as cal_file:
        cal_file.write("bin_lo,bin_hi,mean_pred,mean_actual,count\n")
        for r in cal_rows:
            cal_file.write(f"{r['bin_lo']:.6f},{r['bin_hi']:.6f},{r['mean_pred']:.6f},{r['mean_actual']:.6f},{r['count']}\n")
    print(f"Wrote {cal_path}")

    # Audit CSV: random sample of rows for label-alignment validation
    if args.audit_out and args.audit_out.strip():
        audit_path = Path(args.audit_out.strip()) if Path(args.audit_out.strip()).is_absolute() else out_path / args.audit_out.strip()
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        n_audit = min(args.audit_n, len(audit_rows))
        if n_audit > 0:
            rng_audit = np.random.default_rng(args.seed)
            indices = rng_audit.choice(len(audit_rows), size=n_audit, replace=False)
            sample_audit = [audit_rows[i] for i in indices]
            # Collect all keys across sample for CSV columns (consistent order: known first, then sorted extra)
            known = ("game_number", "map_index", "round_number", "y", "phase", "clamp_reason", "score_raw", "p_unshaped", "p_hat_final")
            all_keys: list[str] = []
            seen: set[str] = set()
            for k in known:
                if any(k in r for r in sample_audit):
                    all_keys.append(k)
                    seen.add(k)
            for r in sample_audit:
                for k in sorted(r.keys()):
                    if k not in seen:
                        all_keys.append(k)
                        seen.add(k)
            with open(audit_path, "w", encoding="utf-8", newline="") as audit_file:
                writer = csv.writer(audit_file)
                writer.writerow(all_keys)
                for r in sample_audit:
                    cells = []
                    for col in all_keys:
                        val = r.get(col, "")
                        if isinstance(val, float) and (val != val or abs(val) >= 1e15):
                            cells.append("")
                        else:
                            cells.append(val if val is not None else "")
                    writer.writerow(cells)
            print(f"Wrote {audit_path} ({n_audit} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
