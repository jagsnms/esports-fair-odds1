#!/usr/bin/env python3
"""
Find GRID titleId for Counter-Strike 2 / CS2 via Central Data GraphQL.
Read-only: reads GRID_API_KEY from env/.env, queries titles and prints matches.
Usage: from repo root, python scripts/grid_find_cs2_title_id.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root and .env
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"

# Central Data endpoint (from adapters/grid_probe/grid_graphql_client.py)
CENTRAL_DATA_GRAPHQL_URL = "https://api-op.grid.gg/central-data/graphql"
ENV_KEY_NAME = "GRID_API_KEY"

# Search terms for CS2 (case-insensitive match on name / nameShortened)
CS2_SEARCH_TERMS = ("counter", "strike", "cs2", "counter-strike-2")
TOP_CANDIDATES = 10


def load_api_key() -> str:
    try:
        from dotenv import load_dotenv
    except ImportError:
        raise SystemExit("pip install python-dotenv")
    load_dotenv(ENV_PATH)
    key = os.environ.get(ENV_KEY_NAME)
    if not key or not str(key).strip():
        raise SystemExit(f"Missing {ENV_KEY_NAME}. Set it in .env or environment.")
    return str(key).strip()


def post_graphql(url: str, query: str, variables: dict | None, api_key: str) -> dict:
    try:
        import requests
    except ImportError:
        raise SystemExit("pip install requests")
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    r = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json", "x-api-key": api_key},
        timeout=30,
    )
    if r.status_code != 200:
        return {"errors": [{"message": f"HTTP {r.status_code}", "body": r.text[:500]}]}
    try:
        return r.json()
    except Exception as e:
        return {"errors": [{"message": f"JSON decode: {e}"}]}


def introspect_titles_field(api_key: str) -> dict | None:
    """Introspect Query type to get 'titles' field args and return type."""
    q = """
    query IntrospectQuery {
      __schema {
        queryType {
          fields {
            name
            args { name type { kind name ofType { kind name } } }
            type { kind name ofType { kind name ofType { kind name } } }
          }
        }
      }
    }
    """
    data = post_graphql(CENTRAL_DATA_GRAPHQL_URL, q, None, api_key)
    if data.get("errors"):
        return None
    fields = (data.get("data") or {}).get("__schema", {}).get("queryType", {}).get("fields") or []
    for f in fields:
        if f.get("name") == "titles":
            return f
    return None


def fetch_titles(api_key: str) -> dict:
    """Query titles with filter (TitleFilter; empty filter to get all). Returns list of Title."""
    q = """
    query Titles($filter: TitleFilter) {
      titles(filter: $filter) {
        id
        name
        nameShortened
      }
    }
    """
    return post_graphql(CENTRAL_DATA_GRAPHQL_URL, q, {"filter": {}}, api_key)


def collect_title_nodes(data: dict) -> list[dict]:
    """Extract list of title nodes from response."""
    err = data.get("errors")
    if err:
        return []
    titles = (data.get("data") or {}).get("titles")
    if not titles and titles is not None and not isinstance(titles, list):
        return []
    if isinstance(titles, list):
        return [t for t in titles if isinstance(t, dict)]
    return []


def matches_cs2(node: dict) -> bool:
    name = (node.get("name") or "").lower()
    short = (node.get("nameShortened") or "").lower()
    for term in CS2_SEARCH_TERMS:
        if term in name or term in short:
            return True
    return False


def main() -> int:
    print("GRID Central Data - find CS2 titleId")
    print(f"Endpoint: {CENTRAL_DATA_GRAPHQL_URL}")
    api_key = load_api_key()
    print("API key loaded from env/.env")

    data = fetch_titles(api_key)
    if data.get("errors"):
        errs = data["errors"]
        print("Query error:", errs[0].get("message", errs))
        return 1
    all_nodes = collect_title_nodes(data)

    if not all_nodes:
        print("No title nodes returned. Introspecting 'titles' field...")
        info = introspect_titles_field(api_key)
        if info:
            print("titles field:", info)
        return 1

    print(f"\nTotal titles fetched: {len(all_nodes)}")
    matches = [n for n in all_nodes if matches_cs2(n)]
    print(f"Matches (name/nameShortened containing any of {CS2_SEARCH_TERMS}): {len(matches)}")
    if matches:
        print("\n--- CS2 candidates ---")
        for n in matches:
            print(f"  id: {n.get('id')!r}  name: {n.get('name')!r}  nameShortened: {n.get('nameShortened')!r}")
        # Clear CS2: prefer nameShortened == "cs2" or "CS2"
        clear = [m for m in matches if (m.get("nameShortened") or "").lower() == "cs2"]
        if clear:
            print("\n--- Clear CS2 match (nameShortened == 'cs2') ---")
            for n in clear:
                print(f"  titleId: {n.get('id')!r}  name: {n.get('name')!r}  nameShortened: {n.get('nameShortened')!r}")

    print(f"\n--- Top {TOP_CANDIDATES} titles (all) ---")
    for n in all_nodes[:TOP_CANDIDATES]:
        print(f"  id: {n.get('id')!r}  name: {n.get('name')!r}  nameShortened: {n.get('nameShortened')!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
