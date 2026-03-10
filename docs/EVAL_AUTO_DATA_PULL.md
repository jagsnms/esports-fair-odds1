# Evaluating Auto Data Pull (step-by-step)

Use this to find where the pipeline fails without changing behavior.

## Pipeline overview

1. **Poller** (runs every 5s): reads `logs/bo3_live_control.json` → fetches BO3 snapshot → normalizes → writes `logs/bo3_live_feed.json`.
2. **App** (on each run): if auto is on and feed file exists → reads feed → overwrites session_state (team names, rounds, map, side, etc.).
3. **Timer**: when auto is on, `st_autorefresh(interval=5000)` triggers a full script rerun every 5s so the app re-reads the feed.

Failure can be: (A) poller not running or not getting live data, (B) feed file not updated or wrong path, (C) app not reading or not applying, (D) app not rerunning so it never re-reads.

---

## Step 1: Run the poller standalone (no app)

This checks whether the poller + API + normalizer work.

1. **Create the control file** (with a **known live** match ID):
   - Open `logs/bo3_live_control.json` (create `logs/` if needed).
   - Put:
     ```json
     {"active": true, "match_id": YOUR_LIVE_MATCH_ID, "team_a_is_team_one": true}
     ```
   - Replace `YOUR_LIVE_MATCH_ID` with a real live match ID (e.g. from the app after you click "Load live matches" and pick a match).

2. **Run the poller from the project root**:
   ```bash
   cd path\to\esports-fair-odds
   set PYTHONPATH=%CD%
   python bo3.gg\poller.py
   ```
   Leave it running. You should see a line in the terminal every ~7s, e.g.:
   - `poller: status=live payload_keys=team_a,team_b,rounds_a,...` → API returned data.
   - `poller: status=empty error=snapshot fetch returned empty` → API returned no snapshot (match not live or API issue).

3. **Wait 15–20 seconds** (2–3 cycles), then stop the poller (Ctrl+C).

4. **Open `logs/bo3_live_feed.json`** and check:
   - `snapshot_status`: `"live"` = had data; `"empty"` = no data.
   - `payload`: if non-empty, you should see `team_a`, `team_b`, `rounds_a`, `rounds_b`, `map_name`, etc.
   - `error`: if set, that’s why the payload is empty.

**Interpretation:**

- If **payload is non-empty** and `snapshot_status` is `"live"`: poller + API + normalizer are fine. The issue is likely app-side (when/how it reads, or timer not rerunning).
- If **payload is always empty** and error is e.g. `"snapshot fetch returned empty"`: either the match is not live on BO3.gg, or the API/adapter is failing. Fix poller/API first; use app diagnostics (Step 2) only after the feed file has live data when run standalone.

---

## Step 2: What the app sees (diagnostics in UI)

When auto data pull is **ON**, open the **"Diagnostics"** sub-section inside the "Auto data pull (BO3.gg)" expander. It shows:

- **Feed file path** – path the app uses to read the feed.
- **File exists?** – whether that file exists.
- **Last modified** – file mtime (so you can see if it’s being updated every ~7s).
- **Payload keys** – keys in `payload` (or "empty" if none).
- **snapshot_status** – from the feed file.
- **error** – from the feed file (if any).

**Interpretation:**

- If **file exists**, **last modified** updates every ~7s, and **payload keys** are non-empty but the main UI (team names, rounds, map) doesn’t update → app read path or session_state apply is wrong (e.g. condition not met, or apply skipped).
- If **last modified** never changes → poller isn’t writing (poller not running, or wrong path, or crashed).
- If **payload keys** is "empty" here but Step 1 showed live data in the file → app might be reading a different path or an old copy; compare path with where the poller writes.

---

## Step 3: Confirm the app reruns every 5s

When auto is on, the app should rerun every 5 seconds (so it re-reads the feed). If it never reruns, you’ll only ever see the first read (often "poller_starting" or stale).

- **Quick check**: Leave the app open on the CS2 In-Play tab with auto ON. Watch the **Diagnostics** "Last modified" time; it should change roughly every 5s. If it never changes, the timer isn’t firing or the script isn’t rerunning.
- If you’re not on the CS2 In-Play tab, the autorefresh may still run (depending on layout); for a clear test, stay on that tab.

---

## Step 4: Common outcomes

| Step 1 (standalone) | Step 2 (diagnostics) | Likely cause |
|--------------------|----------------------|---------------|
| Payload live       | Payload empty        | App reading wrong file or before poller writes; or poller writes elsewhere. |
| Payload live       | Payload live, UI not updating | Session_state not applied (e.g. widget order, condition), or UI bound to different keys. |
| Payload empty      | (any)                | Match not live or API/normalizer issue. Fix poller/API first. |
| (not run)          | Last modified static | Poller not running or not writing to the path the app uses. |

---

## Files involved

- **Control**: `logs/bo3_live_control.json` – `active`, `match_id`, `team_a_is_team_one`. App writes this when you Activate/Deactivate; poller reads it.
- **Feed**: `logs/bo3_live_feed.json` – poller writes; app reads. Contains `timestamp`, `payload`, `snapshot_status`, optional `error`.
- **Poller**: `bo3.gg/poller.py` – run with `python bo3.gg/poller.py` from project root with `PYTHONPATH` set.

After you run Steps 1–2 and have the outcomes (e.g. "Step 1: payload live; Step 2: payload live but UI not updating"), we can target the fix.
