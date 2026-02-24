# GRID PROBE V1 — Reusable GraphQL client for GRID Open Access API.
"""
Minimal client: load API key from .env, POST GraphQL with x-api-key header.
Uses requests and python-dotenv. No modifications to app or BO3 pipeline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# GRID PROBE V1 — endpoint URLs (easy to change)
BASE_URL = "https://api-op.grid.gg"
CENTRAL_DATA_GRAPHQL_URL = "https://api-op.grid.gg/central-data/graphql"
SERIES_STATE_GRAPHQL_URL = "https://api-op.grid.gg/live-data-feed/series-state/graphql"

ENV_KEY_NAME = "GRID_API_KEY"
ENV_RECORD_JSONL = "GRID_RECORD_JSONL"


def load_api_key(env_path: Optional[Path] = None) -> str:
    """Load GRID API key from .env. Raises ValueError if missing. GRID PROBE V1."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        raise ImportError("python-dotenv is required. pip install python-dotenv")
    if env_path and env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
    key = os.environ.get(ENV_KEY_NAME)
    if not key or not str(key).strip():
        raise ValueError(
            f"Missing {ENV_KEY_NAME}. Add GRID_API_KEY=your_key to .env (project root or adapters/grid_probe/)."
        )
    return str(key).strip()


def post_graphql(
    url: str,
    query: str,
    variables: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    timeout: int = 15,
    record_jsonl_path: Optional[Path] = None,
    record_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    POST a GraphQL request. Returns parsed JSON dict.
    Handles: missing key, timeout, non-200, invalid JSON, GraphQL "errors" array.
    When record_jsonl_path is set (or GRID_RECORD_JSONL env), appends one JSONL line per call (no secrets).
    GRID PROBE V1.
    """
    if not api_key or not api_key.strip():
        raise ValueError("API key is required for post_graphql")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }
    payload: Dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables

    record_path: Optional[Path] = record_jsonl_path
    if record_path is None:
        env_path = os.environ.get(ENV_RECORD_JSONL)
        if env_path and str(env_path).strip():
            record_path = Path(str(env_path).strip())

    status_code: Optional[int] = None
    response_data: Dict[str, Any] = {}
    ok = False

    try:
        import requests
    except ImportError:
        raise ImportError("requests is required. pip install requests")
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        status_code = r.status_code
        if r.status_code != 200:
            response_data = {
                "errors": [
                    {"message": f"HTTP {r.status_code}", "body": r.text[:500]}
                ]
            }
        else:
            try:
                response_data = r.json()
            except json.JSONDecodeError as e:
                response_data = {"errors": [{"message": f"Invalid JSON: {e}", "body": r.text[:500]}]}
            ok = not (isinstance(response_data, dict) and response_data.get("errors"))
    except requests.exceptions.Timeout:
        response_data = {"errors": [{"message": "Request timed out"}]}
    except requests.exceptions.RequestException as e:
        response_data = {"errors": [{"message": str(e)}]}

    if isinstance(response_data, dict) and response_data.get("errors"):
        print("[GRID PROBE V1] GraphQL returned errors:", response_data.get("errors"))

    if record_path:
        try:
            from fair_odds.record_jsonl import append_jsonl, utc_iso_z
            rec = {
                "ts_utc": utc_iso_z(),
                "source": "GRID",
                "label": record_label,
                "url": url,
                "variables": variables,
                "query": query,
                "timeout_s": timeout,
                "status_code": status_code,
                "ok": ok,
                "response": response_data,
            }
            append_jsonl(record_path, rec)
        except Exception:
            pass

    return response_data
