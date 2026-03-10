# GRID PROBE V1 — Standalone CS2 probe (Open Access GraphQL)

This is a **parallel scaffold** for testing GRID Open Access API. It does **not** modify the main app or the BO3 pipeline.

## Requirements

- Python 3.9+
- `requests`
- `python-dotenv`

Install if needed:

```bash
pip install requests python-dotenv
```

## .env and API key

- Create a `.env` file in the **project root** (or in `adapters/grid_probe/`).
- Set the GRID API key:

  ```
  GRID_API_KEY=your_key_here
  ```

- The probe **never** prints the API key.

## Running the probe

From the **project root**:

```bash
python -m adapters.grid_probe.probe_grid_cs2
```

Or (with `PYTHONPATH` set so imports work):

```bash
python adapters/grid_probe/probe_grid_cs2.py
```

## What it does

1. **Central Data** — Runs a GraphQL query to find CS2 series and saves the response to  
   `adapters/grid_probe/raw_grid_central_data.json`.
2. **Series State** — If a series ID is found (or `MANUAL_SERIES_ID` is set), runs a Series State query and saves the response to  
   `adapters/grid_probe/raw_grid_series_state.json`.
3. Prints short debug summaries (top-level keys, errors, series IDs, `valid`/`updatedAt` if present).

## Config (top of `probe_grid_cs2.py`)

- `MANUAL_SERIES_ID` — Set to a known series ID to skip Central Data or force a specific series.
- `USE_MANUAL_SERIES_ID_FIRST` — If `True`, use `MANUAL_SERIES_ID` only (do not run Central Data for discovery).
- `SAVE_PRETTY_JSON` — `True` to indent saved JSON for inspection.

## Inspecting saved JSON

To print a compact structure (keys and types to limited depth):

```bash
python -m adapters.grid_probe.inspect_grid_json raw_grid_series_state.json
```

Default file if none given: `raw_grid_series_state.json`.

## Schema introspection (one request per run)

To fetch and save root Query field names for either endpoint (no retries, no loops):

```bash
python -m adapters.grid_probe.introspect_grid_schema --central
python -m adapters.grid_probe.introspect_grid_schema --series-state
```

Outputs:

- `--central` → `raw_grid_central_schema_root.json` and prints root field names.
- `--series-state` → `raw_grid_series_state_schema_root.json` and prints root field names.

## Field introspection (args + return type)

To inspect a specific root field (arguments, return type, and one level of return-type fields), one request per run (plus one optional request for return-type fields):

```bash
python -m adapters.grid_probe.introspect_grid_field --central --field allSeries
python -m adapters.grid_probe.introspect_grid_field --series-state --field seriesState
```

Outputs:

- `--central --field allSeries` → `raw_grid_central_allSeries_introspection.json` and prints args, return type, and likely query shape.
- `--series-state --field seriesState` → `raw_grid_seriesState_introspection.json` and prints args, return type, and likely query shape.

No retries, no loops.

## Endpoints (Open Access)

- Base: `https://api-op.grid.gg`
- Central Data GraphQL: `https://api-op.grid.gg/central-data/graphql`
- Live Series State GraphQL: `https://api-op.grid.gg/live-data-feed/series-state/graphql`

Auth header: `x-api-key: <GRID_API_KEY>`.

## Schema notes

Query strings in `grid_queries.py` are placeholders. If the API returns GraphQL errors, adjust field names and shapes to match the real GRID schema (e.g. via Playground or docs). The probe is built so you can edit the query strings and re-run without changing the rest of the pipeline.
