from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from backend.api.routes_market import get_market_status as get_market_status_route
from backend.deps import set_runner, set_store
from backend.services.runner import Runner
from backend.store.memory_store import MemoryStore
from engine.models import Config


async def test_market_status_reports_no_selected_ticker() -> None:
    store = MemoryStore(max_history=10)
    runner = Runner(store=store, broadcaster=MagicMock())

    status = runner.get_market_status(Config())

    assert status["market_enabled"] is True
    assert status["market_selected_ticker"] is None
    assert status["market_polling_active"] is False
    assert status["market_polling_inactive_reason"] == "no_selected_ticker"
    assert status["market_quote_status"] == "inactive"
    assert status["market_quote_available"] is False


def test_market_status_reports_no_selected_ticker_sync() -> None:
    asyncio.run(test_market_status_reports_no_selected_ticker())


async def test_market_status_reports_ready_when_buffered_quote_exists() -> None:
    store = MemoryStore(max_history=10)
    runner = Runner(store=store, broadcaster=MagicMock())
    runner._market_buffer.push({
        "ts_epoch": 1000.0,
        "bid": 0.44,
        "ask": 0.46,
        "mid": 0.45,
        "ticker": "KXTEST-YES",
    })

    status = runner.get_market_status(Config(kalshi_ticker="KXTEST-YES"))

    assert status["market_selected_ticker"] == "KXTEST-YES"
    assert status["market_polling_active"] is True
    assert status["market_polling_inactive_reason"] is None
    assert status["market_quote_status"] == "ready"
    assert status["market_quote_available"] is True
    assert status["market_buffer_size"] == 1


def test_market_status_reports_ready_when_buffered_quote_exists_sync() -> None:
    asyncio.run(test_market_status_reports_ready_when_buffered_quote_exists())


async def test_market_status_route_returns_runner_visibility_contract() -> None:
    store = MemoryStore(max_history=10)
    await store.update_config({"kalshi_ticker": "KXTEST-YES"})
    runner = Runner(store=store, broadcaster=MagicMock())
    set_store(store)
    set_runner(runner)

    status = await get_market_status_route()

    assert status["market_selected_ticker"] == "KXTEST-YES"
    assert status["market_polling_active"] is True
    assert status["market_polling_inactive_reason"] is None
    assert status["market_quote_status"] == "awaiting_quote"


def test_market_status_route_returns_runner_visibility_contract_sync() -> None:
    asyncio.run(test_market_status_route_returns_runner_visibility_contract())
