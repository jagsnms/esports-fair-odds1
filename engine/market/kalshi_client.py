"""
Kalshi market client: URL resolution (event -> team markets) and bid/ask fetch.

Ported from app35_ml.py: _kalshi_parse_event_ticker, kalshi_list_markets_for_event,
_try_extract_kalshi_ticker, fetch_kalshi_bid_ask (market endpoint + orderbook fallback).
No Streamlit/caching; sync HTTP via requests.
"""
from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

_KALSHI_TICKER_RE = re.compile(r"^[A-Z0-9_.-]{3,80}$", re.IGNORECASE)
_KALSHI_BASES = [
    "https://api.kalshi.com/trade-api/v2",
    "https://api.elections.kalshi.com/trade-api/v2",
]


def _try_extract_kalshi_ticker(raw: str) -> str:
    """Best-effort extraction of a Kalshi market ticker from a pasted URL or raw input."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if _KALSHI_TICKER_RE.match(raw):
        return raw.upper() if raw.isascii() else raw
    try:
        u = urlparse(raw)
        path_parts = [p for p in (u.path or "").split("/") if p]
        for i, part in enumerate(path_parts[:-1]):
            if part.lower() in ("markets", "market", "trade", "event"):
                cand = path_parts[i + 1]
                if _KALSHI_TICKER_RE.match(cand):
                    return cand.upper() if cand.isascii() else cand
        if path_parts:
            cand = path_parts[-1]
            if _KALSHI_TICKER_RE.match(cand):
                return cand.upper() if cand.isascii() else cand
    except Exception:
        pass
    return ""


def _kalshi_parse_event_ticker(url_or_ticker: str) -> str | None:
    """Parse Kalshi event_ticker from a page URL or market ticker.
    - URL like .../kxvalorantgame-26jan29blgfpx -> KXVALORANTGAME-26JAN29BLGFPX
    - Market ticker KXVALORANTGAME-26JAN29BLGFPX-BLG -> KXVALORANTGAME-26JAN29BLGFPX
    """
    if not url_or_ticker:
        return None
    s = str(url_or_ticker).strip()
    if not s:
        return None
    if "://" in s:
        try:
            parsed = urlparse(s)
            path = (parsed.path or "").rstrip("/")
            if path:
                s = path.split("/")[-1]
        except Exception:
            pass
    s = s.strip().strip("/")
    if not s:
        return None
    s_up = s.upper()
    if re.match(r"^[A-Z0-9]+(?:-[A-Z0-9]+)*-[A-Z]{2,6}$", s_up):
        parts = s_up.split("-")
        if len(parts) >= 2:
            return "-".join(parts[:-1])
    return s_up


def resolve_kalshi_match_url(url: str) -> dict[str, Any]:
    """
    Resolve a Kalshi event/game URL to structured options for the frontend.

    Returns:
        {
            "options": [{"key": str, "label": str, "ticker_yes": str}, ...],
            "suggested": str | None  # first option key if any
        }
    """
    if not requests:
        raise RuntimeError("requests is required for Kalshi client")
    ev = _kalshi_parse_event_ticker(url)
    if not ev:
        raise ValueError("Missing or invalid Kalshi URL/event ticker")
    last_err: Exception | None = None
    for base in _KALSHI_BASES:
        try:
            r = requests.get(f"{base}/markets", params={"event_ticker": ev}, timeout=6)
            r.raise_for_status()
            data = r.json() or {}
            markets = data.get("markets", []) or []
            out = []
            for m in markets:
                ticker = m.get("ticker")
                if not ticker:
                    continue
                title = m.get("title") or ""
                subtitle = m.get("subtitle") or ""
                label = f"{title}" + (f" — {subtitle}" if subtitle else "")
                if not label.strip():
                    label = ticker
                out.append({"key": ticker, "label": label, "ticker_yes": ticker})
            out.sort(key=lambda x: (x["key"], x.get("label", "")))
            if not out:
                raise RuntimeError("No markets returned for that event")
            return {
                "options": out,
                "suggested": out[0]["key"] if out else None,
            }
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Kalshi market list failed: {last_err}")


def fetch_kalshi_bid_ask(ticker: str) -> tuple[float, float, float, float]:
    """
    Fetch best bid/ask (YES side) for a Kalshi market.
    Tries GET /markets/{ticker} then GET /markets/{ticker}/orderbook.

    Returns:
        (bid, ask, mid, ts_epoch) with bid/ask/mid in 0..1; ts_epoch is time.time() at fetch.
    """
    if not requests:
        raise RuntimeError("requests is required for Kalshi client")
    t = _try_extract_kalshi_ticker(ticker)
    if not t:
        raise ValueError("Missing Kalshi ticker")

    def _to_int(x: Any) -> int | None:
        if x is None:
            return None
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return None

    def _best_bid(levels: list) -> tuple[float | None, Any]:
        best_p: float | None = None
        best_sz = None
        if not levels:
            return None, None
        for lvl in levels:
            p, q = None, None
            if isinstance(lvl, (list, tuple)) and len(lvl) >= 1:
                p, q = lvl[0], lvl[1] if len(lvl) >= 2 else None
            elif isinstance(lvl, dict):
                p = lvl.get("price")
                q = lvl.get("quantity", lvl.get("size"))
            if p is None:
                continue
            try:
                p_f = float(p)
            except (TypeError, ValueError):
                continue
            p_cents = (p_f * 100.0) if p_f <= 1.0 else p_f
            if best_p is None or p_cents > best_p:
                best_p = p_cents
                best_sz = float(q) if q is not None else None
        return best_p, best_sz

    last_err: Exception | None = None
    for base in _KALSHI_BASES:
        try:
            url_m = f"{base}/markets/{t}"
            r = requests.get(url_m, timeout=5)
            r.raise_for_status()
            data = r.json()
            market = data.get("market", data)
            yb = _to_int(market.get("yes_bid", market.get("yesBid")))
            ya = _to_int(market.get("yes_ask", market.get("yesAsk")))
            nb = _to_int(market.get("no_bid", market.get("noBid")))
            if ya is None and nb is not None:
                ya = 100 - nb
            if yb is not None or ya is not None:
                bid = (yb / 100.0) if yb is not None else None
                ask = (ya / 100.0) if ya is not None else None
                if bid is None:
                    bid = ask if ask is not None else 0.5
                if ask is None:
                    ask = bid
                mid = 0.5 * (float(bid) + float(ask))
                return (float(bid), float(ask), mid, time.time())
        except Exception as e:
            last_err = e
        try:
            url = f"{base}/markets/{t}/orderbook"
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            data = r.json()
            ob = data.get("orderbook", data)
            yes_bids = ob.get("yes", []) or ob.get("yes_bids", []) or ob.get("yesOrders", [])
            no_bids = ob.get("no", []) or ob.get("no_bids", []) or ob.get("noOrders", [])
            best_yes_c, _ = _best_bid(yes_bids)
            best_no_c, _ = _best_bid(no_bids)
            if best_yes_c is None and best_no_c is None:
                raise RuntimeError("No orderbook levels returned")
            bid = (best_yes_c / 100.0) if best_yes_c is not None else None
            ask = ((100.0 - best_no_c) / 100.0) if best_no_c is not None else None
            if bid is None:
                bid = float(ask) if ask is not None else 0.5
            if ask is None:
                ask = float(bid)
            mid = 0.5 * (float(bid) + float(ask))
            return (float(bid), float(ask), mid, time.time())
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Kalshi fetch failed for {t}: {last_err}")
