from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.market import kalshi_client


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_fetch_kalshi_bid_ask_parses_market_dollars_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(url: str, timeout: int = 0):  # noqa: ARG001
        assert url.endswith("/markets/KXTEST-YES")
        return _Response(
            {
                "market": {
                    "yes_bid_dollars": "0.56",
                    "yes_ask_dollars": "0.58",
                }
            }
        )

    monkeypatch.setattr(kalshi_client, "requests", SimpleNamespace(get=_fake_get))

    bid, ask, mid, ts_epoch = kalshi_client.fetch_kalshi_bid_ask("KXTEST-YES")

    assert bid == pytest.approx(0.56)
    assert ask == pytest.approx(0.58)
    assert mid == pytest.approx(0.57)
    assert ts_epoch > 0


def test_fetch_kalshi_bid_ask_derives_yes_bid_from_no_ask_dollars(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(url: str, timeout: int = 0):  # noqa: ARG001
        assert url.endswith("/markets/KXTEST-YES")
        return _Response(
            {
                "market": {
                    "yes_ask_dollars": "0.58",
                    "no_ask_dollars": "0.44",
                }
            }
        )

    monkeypatch.setattr(kalshi_client, "requests", SimpleNamespace(get=_fake_get))

    bid, ask, mid, _ = kalshi_client.fetch_kalshi_bid_ask("KXTEST-YES")

    assert bid == pytest.approx(0.56)
    assert ask == pytest.approx(0.58)
    assert mid == pytest.approx(0.57)


def test_fetch_kalshi_bid_ask_parses_orderbook_fp_dollars_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(url: str, timeout: int = 0):  # noqa: ARG001
        if url.endswith("/markets/KXTEST-YES"):
            return _Response({"market": {}})
        if url.endswith("/markets/KXTEST-YES/orderbook"):
            return _Response(
                {
                    "orderbook_fp": {
                        "yes_dollars": [["0.15", "100.00"]],
                        "no_dollars": [["0.20", "50.00"]],
                    }
                }
            )
        raise AssertionError(url)

    monkeypatch.setattr(kalshi_client, "requests", SimpleNamespace(get=_fake_get))

    bid, ask, mid, _ = kalshi_client.fetch_kalshi_bid_ask("KXTEST-YES")

    assert bid == pytest.approx(0.15)
    assert ask == pytest.approx(0.80)
    assert mid == pytest.approx(0.475)


def test_fetch_kalshi_bid_ask_fails_truthfully_for_empty_orderbook(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(url: str, timeout: int = 0):  # noqa: ARG001
        if url.endswith("/markets/KXTEST-YES"):
            return _Response({"market": {}})
        if url.endswith("/markets/KXTEST-YES/orderbook"):
            return _Response({"orderbook_fp": {"yes_dollars": [], "no_dollars": []}})
        raise AssertionError(url)

    monkeypatch.setattr(kalshi_client, "requests", SimpleNamespace(get=_fake_get))

    with pytest.raises(RuntimeError, match="No orderbook levels returned"):
        kalshi_client.fetch_kalshi_bid_ask("KXTEST-YES")


def test_fetch_kalshi_bid_ask_fails_truthfully_for_one_sided_orderbook(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(url: str, timeout: int = 0):  # noqa: ARG001
        if url.endswith("/markets/KXTEST-YES"):
            return _Response({"market": {}})
        if url.endswith("/markets/KXTEST-YES/orderbook"):
            return _Response({"orderbook_fp": {"yes_dollars": [["0.15", "100.00"]], "no_dollars": []}})
        raise AssertionError(url)

    monkeypatch.setattr(kalshi_client, "requests", SimpleNamespace(get=_fake_get))

    with pytest.raises(RuntimeError, match="Orderbook did not contain both yes and no bid levels"):
        kalshi_client.fetch_kalshi_bid_ask("KXTEST-YES")
