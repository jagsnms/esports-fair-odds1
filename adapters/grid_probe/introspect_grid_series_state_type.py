# GRID PROBE V1 — Introspect a type on the Series State endpoint (e.g. Title).
"""
Run from project root. One request. No retries, no loops.
Series State endpoint only.

  python -m adapters.grid_probe.introspect_grid_series_state_type Title

Saves: raw_grid_seriesState_type_<TypeName>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# GRID PROBE V1
PROBE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROBE_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# GRID PROBE V1
INTROSPECT_TYPE_QUERY = """
query IntrospectType($typeName: String!) {
  __type(name: $typeName) {
    name
    kind
    fields {
      name
      type {
        kind
        name
        ofType { kind name }
      }
    }
  }
}
"""


def _save_json(path: Path, data: dict) -> None:
    # GRID PROBE V1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[GRID PROBE V1] Saved: {path}")


def main() -> None:
    # GRID PROBE V1
    type_name = sys.argv[1] if len(sys.argv) > 1 else "Title"
    out_path = PROBE_DIR / f"raw_grid_seriesState_type_{type_name}.json"

    if str(PROBE_DIR) not in sys.path:
        sys.path.insert(0, str(PROBE_DIR))
    from grid_graphql_client import SERIES_STATE_GRAPHQL_URL, load_api_key, post_graphql

    print("[GRID PROBE V1] Loading API key from .env ...")
    try:
        api_key = load_api_key(ENV_PATH)
    except Exception as e:
        print(f"[GRID PROBE V1] Failed to load API key: {e}")
        sys.exit(1)

    print(f"[GRID PROBE V1] Series State — introspecting type '{type_name}' ...")
    data = post_graphql(
        SERIES_STATE_GRAPHQL_URL,
        INTROSPECT_TYPE_QUERY.strip(),
        variables={"typeName": type_name},
        api_key=api_key,
    )

    if data.get("errors"):
        print("[GRID PROBE V1] GraphQL errors:", data.get("errors"))
        sys.exit(1)

    type_obj = (data.get("data") or {}).get("__type")
    if not type_obj:
        print("[GRID PROBE V1] No __type in response.")
        sys.exit(1)

    _save_json(out_path, type_obj)
    fields = type_obj.get("fields") or []
    print(f"[GRID PROBE V1] Type {type_name} fields:")
    for f in fields:
        if isinstance(f, dict) and f.get("name"):
            t = f.get("type") or {}
            name = t.get("name") or t.get("kind") or "?"
            print(f"  {f['name']}: {name}")


if __name__ == "__main__":
    main()
