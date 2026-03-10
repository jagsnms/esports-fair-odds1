# scraper.py — HLTV team recent matches scraper (shallow + optional deep per-match pages)
# Python 3.9+
# pip install playwright
# python -m playwright install

import sys
import json
import re
import time
import argparse
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE = "https://www.hltv.org"

def log(msg):
    print(msg, flush=True)

def click_cookie_banner(page):
    # HLTV cookie button often says "Allow all cookies"
    try:
        btn = page.locator("button:has-text('Allow all cookies')")
        if btn.count():
            btn.first.click(timeout=4000)
            page.wait_for_timeout(250)
            log("[cookie] allowed all cookies")
    except Exception:
        pass

def wait_cf_verification(page, max_wait_ms=20000):
    """
    If Cloudflare "verify you are human" shows, wait a bit for the user to solve it manually.
    We don't try to auto-solve; we just pause so a headed window can be clicked.
    """
    try:
        challenge = page.locator("text=verify you are human").first
        box = page.locator("iframe[title*='security challenge'], iframe[src*='challenges']")
        start = time.time()
        while (challenge.count() or box.count()) and (time.time() - start) * 1000 < max_wait_ms:
            page.wait_for_timeout(500)
        if (challenge.count() or box.count()):
            log("[cf] still present after wait; continuing anyway")
    except Exception:
        pass

def goto_with_cf(page, url, wait_selector=None, timeout=15000):
    page.goto(url, wait_until="domcontentloaded")
    click_cookie_banner(page)
    wait_cf_verification(page)
    if wait_selector:
        try:
            page.wait_for_selector(wait_selector, timeout=timeout)
        except PWTimeout:
            # try a nudge scroll
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight/3)")
            except Exception:
                pass
            if wait_selector:
                page.wait_for_selector(wait_selector, timeout=timeout)

def extract_int_safe(text):
    try:
        return int(str(text).strip())
    except Exception:
        return None

def scrape_team_rows(page, limit):
    """
    Returns a list of dicts for recent matches from the team page:
      {opponent, win, series_us, series_them, match_href}
    NOTE: this only uses what's visible on the team matches tab.

    IMPORTANT: Rows with missing series scores (typical for "Upcoming") are SKIPPED.
    """
    out = []

    # Open matches tab
    try:
        tab_btn = page.locator("[data-content-id='matchesBox']")
        if tab_btn.count():
            tab_btn.first.click()
            page.wait_for_timeout(800)
    except Exception:
        pass

    # Scroll a bit to ensure rows render
    for y in (600, 1200, 1800):
        try:
            page.evaluate(f"window.scrollTo(0,{y})")
            page.wait_for_timeout(200)
        except Exception:
            pass

    # Newer HLTV layouts use a mixture of row classes. Try primary then fallbacks.
    rows = page.locator("#matchesBox .team-row")
    if rows.count() == 0:
        rows = page.locator("#matchesBox .result-con, #matchesBox .result")
    count = rows.count()

    for i in range(min(count, limit)):
        row = rows.nth(i)

        # Opponent
        opp_el = row.locator(".team-name.team-2")
        if opp_el.count() == 0:
            # fallback: generic
            opp_el = row.locator(".team-name").last
        opponent = (opp_el.first.inner_text().strip() if opp_el.count() else "").strip()
        if not opponent:
            # try a different structure if needed
            opp_el2 = row.locator("a.team")  # last link often opponent
            if opp_el2.count():
                opponent = opp_el2.last.inner_text().strip()

        # Series score (the big numbers next to each other)
        sc = row.locator(".score-cell .score")
        series_us = series_them = None
        if sc.count() >= 2:
            series_us = extract_int_safe(sc.nth(0).inner_text())
            series_them = extract_int_safe(sc.nth(1).inner_text())
        else:
            # fallback: find two numbers in the row text
            txt = row.inner_text()
            m = re.findall(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b", txt)
            if m:
                try:
                    a, b = m[0]
                    series_us = extract_int_safe(a)
                    series_them = extract_int_safe(b)
                except Exception:
                    pass

        # --- SKIP upcoming/uns cored rows (prevents false losses from future matches) ---
        if series_us is None or series_them is None:
            # log for visibility but don't add to list
            log(f"[skip] missing series score (likely upcoming): opp='{opponent}'")
            continue

        win = series_us > series_them

        # Match page link
        link = row.locator("a.matchpage-button")
        match_href = ""
        if link.count():
            href = link.first.get_attribute("href") or ""
            if href:
                match_href = urljoin(BASE, href)

        out.append({
            "opponent": opponent or "Unknown",
            "win": bool(win),
            "series_us": series_us,
            "series_them": series_them,
            "match_href": match_href
        })

    return out

def parse_per_map_scores(text_blob):
    """
    From a match-page text blob, extract per-map scores like 16-9, 13-16, 19-17 (OT).
    Returns list of dicts: [{us, them, ot: bool}]
    We cannot always tell which side is 'us' vs 'them' without team alignment,
    but since the team page already told us the series winner, we can
    position via series_us/series_them if needed later. For now we return pairs.
    """
    maps = []
    # capture '16-12', '16 - 12', optional '(OT)' marker either immediately or nearby
    # We'll get matches and post-check local OT tokens.
    pattern = re.compile(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b", re.I)
    for m in pattern.finditer(text_blob):
        us = extract_int_safe(m.group(1))
        them = extract_int_safe(m.group(2))
        if us is None or them is None:
            continue
        # Look around the match for "OT"
        start, end = m.span()
        window = text_blob[max(0, start-20):min(len(text_blob), end+20)]
        ot = ("OT" in window.upper()) or ("Overtime" in window)
        maps.append({"us": us, "them": them, "ot": bool(ot)})
    return maps

def scrape_match_page(context, match_url):
    """
    Visit a match page and pull a rough list of per-map scores.
    We keep this robust by regexing text; HLTV changes DOM often.
    """
    page2 = context.new_page()
    try:
        goto_with_cf(page2, match_url)
        # give it a moment for dynamic sections
        page2.wait_for_timeout(800)

        # Collect full visible text (robust against layout changes)
        try:
            txt = page2.locator("body").inner_text()
        except Exception:
            txt = page2.content()

        per_map = parse_per_map_scores(txt)
        return per_map
    finally:
        try:
            page2.close()
        except Exception:
            pass

def fetch_last_matches(team_id, team_slug, limit=15, deep=False, max_deep=5, headed=False):
    url = f"{BASE}/team/{team_id}/{team_slug}#tab-matchesBox"
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        context = browser.new_context(
            viewport={"width": 1400, "height": 900},
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
            locale="en-US",
        )
        # Mask webdriver
        context.add_init_script("""() => { Object.defineProperty(navigator, 'webdriver', {get: () => undefined}); }""")
        page = context.new_page()

        goto_with_cf(page, url, wait_selector="#matchesBox")
        # Light jiggle to ensure pagination/rows populate
        try:
            for _ in range(2):
                page.mouse.wheel(0, 1200)
                page.wait_for_timeout(200)
        except Exception:
            pass

        rows = scrape_team_rows(page, limit=limit)

        # Deep scrape (optional): open match pages in order to get per-map scores
        if deep:
            opened = 0
            for r in rows:
                if opened >= max_deep:
                    break
                href = r.get("match_href", "")
                if not href:
                    continue
                try:
                    per_map = scrape_match_page(context, href)
                    # attach in-place
                    r["maps"] = per_map
                except Exception as e:
                    r["maps"] = []
                opened += 1

        browser.close()
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("team_id", type=str)
    ap.add_argument("team_slug", type=str)
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--deep", type=int, default=0, help="1=visit match pages to extract per-map scores")
    ap.add_argument("--max-deep", type=int, default=5, help="max match pages to open (ordered)")
    ap.add_argument("--headed", action="store_true", help="show browser (for manual CF challenges)")
    args = ap.parse_args()

    data = fetch_last_matches(
        args.team_id,
        args.team_slug,
        limit=args.limit,
        deep=(args.deep == 1),
        max_deep=args.max_deep,   # fixed: Namespace attribute access
        headed=args.headed
    )
    # Print only the JSON array as last line (app expects this)
    print(json.dumps(data, ensure_ascii=False))

if __name__ == "__main__":
    main()
