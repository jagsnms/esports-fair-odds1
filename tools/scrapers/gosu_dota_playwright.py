# gosu_dota_scraper.py — event-driven, resilient, bottom-paginator-only, fuzzy team match
# Adds zoom control and forces UTF-8 stdout to avoid Windows encoding errors.

import re, json, argparse, sys, asyncio, time, unicodedata, string
from typing import List, Dict, Optional
from difflib import SequenceMatcher

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from playwright._impl._errors import TargetClosedError

# ---- Force UTF-8 stdout on Windows consoles / subprocess pipes ----
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- Windows asyncio fix (lets Playwright spawn properly under Streamlit) ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

BASE = "https://www.gosugamers.net"

# Finished match row anchors only
SEL_MATCH_ROWS = "li.MuiListItem-root a.MuiButtonBase-root[href*='/dota2/tournaments/'][href*='/matches/']"

# Column xpaths within each row anchor
X_LEFT_P = ".//div[contains(@class,'MuiGrid-container')]/div[1]//p"
X_RIGHT_P = ".//div[contains(@class,'MuiGrid-container')]/div[3]//p"
X_SCORE_DIGITS = ".//div[contains(@class,'MuiGrid-container')]/div[2]//span[contains(@class,'MuiTypography-p5')]"

# Safety rails
HARD_MAX_PAGES = 25
NO_NEW_MATCH_PAGES_STOP = 2

def dprint(debug: bool, msg: str):
    if debug:
        print(msg, flush=True)

# --- normalization + fuzzy match ---
_ZWS = "\u200b\u200c\u200d"
def _clean_spaces(s: str) -> str:
    s = s.replace("\u00a0", " ").replace(_ZWS, "")
    return re.sub(r"\s+", " ", s).strip()

def norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("team ", " ")
    s = s.replace("_", " ")
    s = s.translate(str.maketrans("", "", string.punctuation))
    return _clean_spaces(s)

def looks_like(a: str, b: str) -> bool:
    if not a or not b: return False
    if a == b: return True
    ta, tb = set(a.split()), set(b.split())
    if ta and tb and (ta.issubset(tb) or tb.issubset(ta)):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.72

def click_cookie_banners(page, debug=False):
    for label in ["Accept", "I agree", "Agree", "Got it", "Accept all", "Allow all", "OK"]:
        try:
            btn = page.get_by_role("button", name=re.compile(label, re.I))
            if btn and btn.count() > 0:
                btn.first.click(timeout=800)
                page.wait_for_timeout(150)
                dprint(debug, f"[debug] cookie button clicked: {label}")
        except Exception:
            pass

# ---------------------------
# Zoom helpers
# ---------------------------
def apply_zoom(page, zoom_pct: int, debug=False):
    """
    Try CSS zoom first; if site ignores it, fall back to real Ctrl+- presses.
    zoom_pct: e.g., 80 => 80% (approx 2 Ctrl+- from 100%)
    """
    zoom_pct = max(50, min(120, int(zoom_pct or 100)))
    try:
        page.evaluate(f"document.documentElement.style.zoom='{zoom_pct}%'; document.body.style.zoom='{zoom_pct}%';")
        page.wait_for_timeout(100)
        dprint(debug, f"[debug] applied CSS zoom {zoom_pct}%")
        return
    except Exception:
        pass
    try:
        steps = max(0, int(round((100 - zoom_pct) / 10.0)))
        page.keyboard.down("Control")
        for _ in range(steps):
            page.keyboard.press("-")
            page.wait_for_timeout(60)
        page.keyboard.up("Control")
        dprint(debug, f"[debug] applied keyboard zoom to ~{zoom_pct}% via Ctrl+'-' x {steps}")
    except Exception:
        pass

# ---------------------------
# Scrolling & paginator utils
# ---------------------------
def scroll_to_bottom_until_stable(page, max_loops=40, settle_ms=300, debug=False):
    last_h = -1
    same = 0
    for i in range(max_loops):
        if page.is_closed():
            return
        try:
            page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
        except Exception:
            return
        try:
            page.wait_for_timeout(settle_ms)
        except Exception:
            return
        try:
            h = page.evaluate("document.documentElement.scrollHeight")
        except Exception:
            return
        dprint(debug, f"[debug] bottom pass {i+1}: h={h}")
        if h == last_h:
            same += 1
            if same >= 2:
                break
        else:
            same = 0
            last_h = h

def wait_for_matches_present(page, timeout_ms=7000, debug=False) -> bool:
    scroll_to_bottom_until_stable(page, debug=debug)
    try:
        page.wait_for_selector(SEL_MATCH_ROWS, timeout=timeout_ms, state="attached")
        return True
    except Exception:
        scroll_to_bottom_until_stable(page, debug=debug)
        try:
            page.wait_for_selector(SEL_MATCH_ROWS, timeout=int(timeout_ms/2), state="attached")
            return True
        except Exception:
            return False

def ensure_matches_visible(page, debug=False) -> bool:
    if not wait_for_matches_present(page, debug=debug):
        dprint(debug, "[debug] matches not present")
        return False
    try:
        first = page.locator(SEL_MATCH_ROWS).first
        first.scroll_into_view_if_needed(timeout=1500)
        page.wait_for_timeout(150)
    except Exception:
        pass
    return page.locator(SEL_MATCH_ROWS).count() > 0

def get_bottom_paginator_group(page, debug=False):
    btns = page.get_by_role("button", name=re.compile(r"^Go to page \d+$"))
    n = btns.count()
    if n == 0:
        return []
    max_y = -1
    ys = []
    for i in range(n):
        try:
            box = btns.nth(i).bounding_box()
        except PWTimeout:
            box = None
        y = box["y"] if box else -1
        ys.append(y)
        if y > max_y:
            max_y = y
    bottom_idxs = [i for i, y in enumerate(ys) if y >= max_y - 80]
    dprint(debug, f"[debug] bottom paginator group size={len(bottom_idxs)}, max_y={max_y}")
    return [btns.nth(i) for i in bottom_idxs]

def bottom_current_page(page, debug=False) -> int:
    candidates = page.locator("button[aria-current='page']")
    n = candidates.count()
    if n == 0:
        return 1
    best_i, best_y = 0, -1
    for i in range(n):
        try:
            box = candidates.nth(i).bounding_box()
        except PWTimeout:
            box = None
        y = box["y"] if box else -1
        if y > best_y:
            best_y, best_i = y, i
    btn = candidates.nth(best_i)
    label = (btn.get_attribute("aria-label") or btn.inner_text() or "").strip()
    m = re.search(r"(\d+)", label)
    cur = int(m.group(1)) if m else 1
    dprint(debug, f"[debug] bottom current page={cur}")
    return cur

def bottom_last_page(page, debug=False) -> int:
    group = get_bottom_paginator_group(page, debug=debug)
    if not group:
        return 1
    nums = []
    for g in group:
        try:
            label = g.get_attribute("aria-label") or ""
            m = re.search(r"(\d+)$", label)
            if m: nums.append(int(m.group(1)))
        except Exception:
            pass
    last = max(nums) if nums else 1
    dprint(debug, f"[debug] bottom last page={last}")
    return last

def click_bottom_page_and_wait(page, target_n: int, debug=False) -> bool:
    if page.is_closed():
        return False
    scroll_to_bottom_until_stable(page, debug=debug)
    group = get_bottom_paginator_group(page, debug=debug)
    if not group:
        dprint(debug, "[debug] no bottom paginator group found")
        return False

    target = None
    for g in group:
        lab = g.get_attribute("aria-label") or ""
        if re.search(rf"Go to page {target_n}$", lab):
            target = g
            break
    if target is None:
        dprint(debug, f"[debug] bottom paginator has no 'page {target_n}' button")
        return False

    # snapshot to detect content swap
    try:
        first_anchor = page.locator(SEL_MATCH_ROWS).first
        prev_href = first_anchor.get_attribute("href")
    except Exception:
        prev_href = None

    try:
        target.scroll_into_view_if_needed(timeout=1500)
        target.click(timeout=3000)
    except Exception as e:
        dprint(debug, f"[debug] click page {target_n} failed: {e}")
        return False

    deadline = time.time() + 12.0
    while time.time() < deadline:
        if page.is_closed():
            return False
        try:
            cur = bottom_current_page(page, debug=False)
            if cur == target_n:
                break
        except Exception:
            pass
        try:
            now_href = page.locator(SEL_MATCH_ROWS).first.get_attribute("href")
            if prev_href and now_href and now_href != prev_href:
                break
        except Exception:
            pass
        try:
            page.wait_for_timeout(150)
        except Exception:
            return False

    ensure_matches_visible(page, debug=debug)
    return True

# -------------
# Row extraction (defensive)
# -------------
def extract_rows(page, debug=False) -> List[Dict]:
    rows = []
    try:
        anchors = page.locator(SEL_MATCH_ROWS)
        count = anchors.count()
    except (TargetClosedError, PWTimeout, Exception):
        return rows
    dprint(debug, f"[debug] anchors on page: {count}")

    for i in range(count):
        if page.is_closed():
            break
        try:
            a = anchors.nth(i)
            href = a.get_attribute("href") or ""
            left_loc = a.locator("xpath=" + X_LEFT_P).first
            right_loc = a.locator("xpath=" + X_RIGHT_P).first
            left_loc.wait_for(state="visible", timeout=1800)
            right_loc.wait_for(state="visible", timeout=1800)
            left = left_loc.inner_text().strip()
            right = right_loc.inner_text().strip()
        except (TargetClosedError, PWTimeout, Exception):
            continue

        s_left = s_right = None
        try:
            spans = a.locator("xpath=" + X_SCORE_DIGITS)
            cnt = spans.count()
            if cnt >= 2:
                try:
                    spans.nth(0).wait_for(state="visible", timeout=1200)
                    spans.nth(1).wait_for(state="visible", timeout=1200)
                    s_left = int(spans.nth(0).inner_text().strip())
                    s_right = int(spans.nth(1).inner_text().strip())
                except Exception:
                    s_left = s_right = None
        except (TargetClosedError, PWTimeout, Exception):
            pass

        rows.append({
            "left_team": left,
            "right_team": right,
            "left_score": s_left,
            "right_score": s_right,
            "match_url": BASE + href if href.startswith("/") else href,
        })
    return rows

# -------
# Scraper
# -------
def scrape_team_recent(
    team_slug: str,
    team_name: str,
    target_matches: int = 14,
    draw_is_win: bool = False,
    headed: bool = True,
    debug: bool = False,
    channel: Optional[str] = None,   # None | 'chrome' | 'msedge'
    zoom: int = 80,                  # page zoom percent
) -> List[Dict]:
    out: List[Dict] = []
    seen_urls = set()
    target_norm = norm(team_name)

    with sync_playwright() as p:
        launch_kwargs = dict(
            headless=not headed,
            slow_mo=100 if headed else 0,
            args=["--disable-blink-features=AutomationControlled"],
        )
        if channel:
            browser = p.chromium.launch(channel=channel, **launch_kwargs)
        else:
            browser = p.chromium.launch(**launch_kwargs)

        ctx = browser.new_context(
            viewport={"width": 1400, "height": 900},
            locale="en-US",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/123.0.0.0 Safari/537.36"),
        )
        page = ctx.new_page()

        page.goto(f"{BASE}/dota2/teams/{team_slug}", wait_until="domcontentloaded")
        click_cookie_banners(page)

        # Apply zoom (CSS or keyboard) so bottom paginator stays within the viewport
        apply_zoom(page, zoom, debug=debug)

        # Ensure matches visible
        if not ensure_matches_visible(page, debug=debug):
            dprint(debug, "[debug] matches not visible after load; aborting")
            ctx.close(); browser.close()
            return out

        # Normalize to page 1 on bottom paginator
        cur = bottom_current_page(page, debug=debug)
        if cur != 1:
            click_bottom_page_and_wait(page, 1, debug=debug)
            cur = bottom_current_page(page, debug=debug)

        last = bottom_last_page(page, debug=debug)
        stagnant = 0
        pages_walked = 1
        no_new_count = 0

        while len(out) < target_matches and not page.is_closed():
            if not ensure_matches_visible(page, debug=debug):
                dprint(debug, "[debug] matches not visible on current page; breaking")
                break

            before = len(out)
            rows = extract_rows(page, debug=debug)
            for r in rows:
                url = r["match_url"]
                if url in seen_urls:
                    continue
                ls, rs = r["left_score"], r["right_score"]
                if ls is None or rs is None:
                    continue

                left_n, right_n = norm(r["left_team"]), norm(r["right_team"])
                if looks_like(left_n, target_norm):
                    opponent = r["right_team"]; win = ls > rs
                elif looks_like(right_n, target_norm):
                    opponent = r["left_team"]; win = rs > ls
                else:
                    continue

                if ls == rs and not draw_is_win:
                    win = False

                out.append({
                    "team": team_name,
                    "opponent": opponent,
                    "win": bool(win),
                    "score": f"{ls}-{rs}",
                    "match_url": url,
                })
                seen_urls.add(url)

                if len(out) >= target_matches:
                    break

            added = len(out) - before
            if added == 0:
                no_new_count += 1
            else:
                no_new_count = 0

            if len(out) >= target_matches:
                break
            if pages_walked >= HARD_MAX_PAGES:
                dprint(debug, f"[debug] hit HARD_MAX_PAGES={HARD_MAX_PAGES}; stopping")
                break
            if no_new_count >= NO_NEW_MATCH_PAGES_STOP:
                dprint(debug, "[debug] no new matches across consecutive pages; stopping")
                break

            prev_cur = cur
            if cur >= last:
                dprint(debug, "[debug] reached last matches page; stopping")
                break

            if not click_bottom_page_and_wait(page, cur + 1, debug=debug):
                dprint(debug, f"[debug] failed to click page {cur+1}; stopping")
                break

            cur = bottom_current_page(page, debug=debug)
            last = bottom_last_page(page, debug=debug)
            pages_walked += 1

            if cur == prev_cur:
                stagnant += 1
                dprint(debug, f"[debug] page number didn’t change (stagnant={stagnant})")
                if stagnant >= 2:
                    break
            else:
                stagnant = 0

        ctx.close(); browser.close()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-slug", required=True, help="e.g., 14009-team-spirit")
    ap.add_argument("--team-name", required=True, help='e.g., "team spirit" (pass slug-derived name)')
    ap.add_argument("--target", type=int, default=14, help="how many finished matches to return")
    ap.add_argument("--headed", action="store_true", help="open a visible browser window")
    ap.add_argument("--draw-is-win", action="store_true", help="treat draws as wins")
    ap.add_argument("--debug", action="store_true", help="print debug info")
    ap.add_argument("--channel", default=None, help="chrome | msedge | None (bundled)")
    ap.add_argument("--zoom", type=int, default=80, help="page zoom percent, e.g. 80 for 80%%")
    args = ap.parse_args()

    data = scrape_team_recent(
        team_slug=args.team_slug,
        team_name=args.team_name,
        target_matches=args.target,
        draw_is_win=args.draw_is_win,
        headed=args.headed,
        debug=args.debug,
        channel=args.channel,
        zoom=args.zoom,
    )

    # Robust UTF-8 output (avoid cp1252 choke)
    try:
        print(json.dumps(data, ensure_ascii=False))
    except Exception:
        out = json.dumps(data, ensure_ascii=False).encode("utf-8", errors="replace")
        sys.stdout.buffer.write(out + b"\n")
        sys.stdout.buffer.flush()

if __name__ == "__main__":
    main()
