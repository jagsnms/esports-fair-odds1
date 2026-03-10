# GRID PROBE V1 — One-time GraphQL introspection for root Query fields.
"""
Run from project root. One request per run. No retries, no loops.

  python -m adapters.grid_probe.introspect_grid_schema --central
  python -m adapters.grid_probe.introspect_grid_schema --series-state

Saves:
  --central       -> raw_grid_central_schema_root.json
  --series-state  -> raw_grid_series_state_schema_root.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# GRID PROBE V1
PROBE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROBE_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

INTROSPECTION_QUERY = """
query IntrospectRootQueryFields {
  __schema {
    queryType {
      name
      fields {
        name
      }
    }
  }
}
"""

# GRID PROBE V1 — output paths
RAW_CENTRAL_SCHEMA_PATH = PROBE_DIR / "raw_grid_central_schema_root.json"
RAW_SERIES_STATE_SCHEMA_PATH = PROBE_DIR / "raw_grid_series_state_schema_root.json"


def _save_json(path: Path, data: dict) -> None:
    # GRID PROBE V1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[GRID PROBE V1] Saved: {path}")


def _print_root_fields(fields: list[dict]) -> None:
    # GRID PROBE V1 — print root Query field names cleanly
    names = [f.get("name") for f in fields if isinstance(f, dict) and f.get("name")]
    if not names:
        print("[GRID PROBE V1] No root fields found.")
        return
    print("[GRID PROBE V1] Root Query fields:")
    for n in sorted(names):
        print(f"  {n}")


def main() -> None:
    # GRID PROBE V1 — endpoint selection (one request per run)
    parser = argparse.ArgumentParser(
        description="GRID PROBE V1: Introspect root Query fields (one request per run)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--central", action="store_true", help="Central Data GraphQL")
    group.add_argument("--series-state", action="store_true", help="Live Data Feed Series State GraphQL")
    args = parser.parse_args()

    if str(PROBE_DIR) not in sys.path:
        sys.path.insert(0, str(PROBE_DIR))
    from grid_graphql_client import (
        CENTRAL_DATA_GRAPHQL_URL,
        SERIES_STATE_GRAPHQL_URL,
        load_api_key,
        post_graphql,
    )

    print("[GRID PROBE V1] Loading API key from .env ...")
    try:
        api_key = load_api_key(ENV_PATH)
    except Exception as e:
        print(f"[GRID PROBE V1] Failed to load API key: {e}")
        sys.exit(1)

    if args.central:
        url = CENTRAL_DATA_GRAPHQL_URL
        out_path = RAW_CENTRAL_SCHEMA_PATH
        print("[GRID PROBE V1] Central Data GraphQL — one introspection request ...")
    else:
        url = SERIES_STATE_GRAPHQL_URL
        out_path = RAW_SERIES_STATE_SCHEMA_PATH
        print("[GRID PROBE V1] Live Data Feed Series State GraphQL — one introspection request ...")

    # GRID PROBE V1 — one request only, no retries
    data = post_graphql(url, INTROSPECTION_QUERY.strip(), api_key=api_key)
    _save_json(out_path, data)

    if data.get("errors"):
        print("[GRID PROBE V1] GraphQL returned errors; see saved file.")
        return

    # Extract and print root field names
    try:
        schema = (data.get("data") or {}).get("__schema") or {}
        query_type = schema.get("queryType") or {}
        fields = query_type.get("fields") or []
        _print_root_fields(fields)
    except Exception as e:
        print(f"[GRID PROBE V1] Could not extract fields: {e}")


if __name__ == "__main__":
    main()
