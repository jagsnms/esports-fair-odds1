"""
Regression: CS2 compute must not use legacy merged econ field (cash_loadout_totals).
micro_adjustment_cs2 should depend on loadout_totals (alive equipment totals) only.
"""
from __future__ import annotations

import unittest

from engine.compute.micro_adj_cs2 import micro_adjustment_cs2
from engine.models import Frame


def test_micro_adj_ignores_cash_loadout_totals_sentinel() -> None:
    """Changing cash_loadout_totals alone must not change micro_adjustment_cs2 output."""
    base = Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=(0, 0),
        alive_counts=(5, 5),
        hp_totals=(500.0, 500.0),
        cash_loadout_totals=(999999.0, 0.0),  # sentinel; must be ignored
        loadout_totals=(12000.0, 6000.0),     # realistic; should be used
    )
    adj1 = micro_adjustment_cs2(base)

    changed_legacy_only = Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=(0, 0),
        alive_counts=(5, 5),
        hp_totals=(500.0, 500.0),
        cash_loadout_totals=(0.0, 999999.0),  # swapped sentinel; must still be ignored
        loadout_totals=(12000.0, 6000.0),     # unchanged
    )
    adj2 = micro_adjustment_cs2(changed_legacy_only)

    assert adj1 == adj2


def test_micro_adj_changes_with_loadout_totals() -> None:
    """Changing loadout_totals should change the econ contribution (if not clipped away)."""
    f1 = Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=(0, 0),
        alive_counts=(5, 5),
        hp_totals=(500.0, 500.0),
        cash_loadout_totals=(999999.0, 0.0),  # irrelevant
        loadout_totals=(12000.0, 6000.0),
    )
    f2 = Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=(0, 0),
        alive_counts=(5, 5),
        hp_totals=(500.0, 500.0),
        cash_loadout_totals=(999999.0, 0.0),  # irrelevant
        loadout_totals=(6000.0, 12000.0),     # flipped advantage
    )
    adj1 = micro_adjustment_cs2(f1)
    adj2 = micro_adjustment_cs2(f2)
    assert adj1 != adj2
    assert adj1 > adj2


class TestNoMergedEconInCs2Compute(unittest.TestCase):
    """Run the same tests via unittest."""

    def test_micro_adj_ignores_cash_loadout_totals_sentinel(self) -> None:
        test_micro_adj_ignores_cash_loadout_totals_sentinel()

    def test_micro_adj_changes_with_loadout_totals(self) -> None:
        test_micro_adj_changes_with_loadout_totals()


if __name__ == "__main__":
    unittest.main()

