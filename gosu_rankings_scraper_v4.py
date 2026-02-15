# gosu_rankings_scraper_v4.py
# Python 3.9+; Playwright required
# pip install playwright
# python -m playwright install

import sys, csv, re, argparse, os
from typing import List, Dict, Optional
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE = "https://www.gosugamers.net"
RANKINGS_URL = f"{BASE}/dota2/rankings"
ROW_SEL = "a[href*='/dota2/teams/']"
PAGER_SEL = "nav[aria-label='pagination navigation']"

def log(msg): 
    print(msg, flush=True)

def clean_team_name(text):
    return re.sub(r"^\s*#?\d+\s*", "", (text or "")).strip()

def parse_slug(href):
    m = re.search(r"/dota2/teams/([^/?#]+)", href or "")
    return m.group(1) if m else ""

def parse_team_id(slug):
    m = re.match(r"(\d+)", slug or "")
    return int(m.group(1)) if m else None

def get_team_name(a):
    try:
        return clean_team_name(a.inner_text().strip())
    except Exception:
        return ""

def get_rank_near(a):
    # Anchor text first
    try:
        txt = a.inner_text().strip()
        m = re.search(r"#?\s*(\d{1,4})\b", txt)
        if m: return int(m.group(1))
    except Exception:
        pass
    # Row-ish parent fallback
    try:
        parent = a.locator(
            "xpath=ancestor::tr | ancestor::li | ancestor::div[contains(@class,'row')] | ancestor::*[contains(@class,'MuiGrid-container')]"
        ).first
        txt2 = parent.inner_text().strip()
        m2 = re.search(r"#?\s*(\d{1,4})\b", txt2)
        return int(m2.group(1)) if m2 else None
    except Exception:
        return None

def get_rating_near(a):
    try:
        parent = a.locator(
            "xpath=ancestor::tr | ancestor::li | ancestor::div[contains(@class,'row')] | ancestor::*[contains(@class,'MuiGrid-container')]"
        ).first
        txt = parent.inner_text().strip()
        # capture ints or floats (e.g., 1234 or 1234.56)
        cands = re.findall(r"(\d{3,5}(?:\.\d{1,2})?)", txt)
        if not cands: return None
        nums = [float(x) for x in cands]
        # ratings are usually the largest number in the row
        return max(nums) if nums else None
    except Exception:
        return None

def click_cookie_banners(page):
    labels = ["Accept", "I agree", "Agree", "Got it", "Accept all", "Allow all", "OK"]
    for label in labels:
        try:
            btn = page.get_by_role("button", name=re.compile(label, re.I))
            if btn.count():
                btn.first.click(timeout=800)
                page.wait_for_timeout(150)
                log(f"[cookie] clicked '{label}'")
                return
        except Exception:
            pass

def jiggle(page):
    # Nudge lazy loaders
    try:
        for _ in range(3):
            page.mouse.wheel(0, 2200); page.wait_for_timeout(180)
        for key in ["PageDown", "End", "End", "Home"]:
            try: page.keyboard.press(key); page.wait_for_timeout(120)
            except: pass
        page.evaluate("""() => { window.dispatchEvent(new Event('resize')); window.dispatchEvent(new Event('scroll')); }""")
        for y in (500, 1500, 3000, 4500, 0):
            page.evaluate(f"window.scrollTo(0, {y})"); page.wait_for_timeout(140)
    except Exception:
        pass

def wait_for_rankings(page):
    try:
        page.wait_for_selector(ROW_SEL, timeout=8000); return True
    except PWTimeout:
        try:
            page.wait_for_selector(PAGER_SEL, timeout=5000)
            jiggle(page)
            return page.locator(ROW_SEL).count() > 0
        except PWTimeout:
            return False

def get_last_page(page):
    btns = page.get_by_role("button", name=re.compile(r"Go to page \d+$"))
    last = 1
    for i in range(btns.count()):
        lab = btns.nth(i).get_attribute("aria-label") or ""
        m = re.search(r"(\d+)$", lab)
        if m: last = max(last, int(m.group(1)))
    return last

def get_current_page(page):
    cur = page.locator("button[aria-current='page']")
    if cur.count():
        lab = (cur.first.get_attribute("aria-label") or "").strip()
        m = re.search(r"(\d+)$", lab)
        if m: return int(m.group(1))
        try:
            t = cur.first.inner_text().strip()
            m2 = re.search(r"(\d+)", t)
            if m2: return int(m2.group(1))
        except Exception: 
            pass
    return 1

def click_page(page, n):
    btn = page.get_by_role("button", name=re.compile(rf"^Go to page {n}$"))
    if not btn.count() or btn.first.is_disabled(): return False
    btn.first.scroll_into_view_if_needed()
    btn.first.click(); page.wait_for_timeout(500)
    return True

def click_next_chevron(page):
    nxt = page.get_by_role("button", name=re.compile(r"Go to next page", re.I))
    if not nxt.count() or nxt.first.is_disabled(): return False
    nxt.first.scroll_into_view_if_needed()
    nxt.first.click(); page.wait_for_timeout(550)
    return True

def click_load_more(page):
    btn = page.get_by_role("button", name=re.compile(r"(Load more|Show more)", re.I))
    if not btn.count() or btn.first.is_disabled(): return False
    btn.first.scroll_into_view_if_needed()
    btn.first.click(); page.wait_for_timeout(800)
    return True

def load_seed_csv(seed_path):
    seen_slugs, seen_ids = set(), set()
    if not seed_path or not os.path.exists(seed_path):
        return seen_slugs, seen_ids
    with open(seed_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s = (row.get("slug") or "").strip()
            if s: seen_slugs.add(s)
            tid = row.get("team_id")
            if tid:
                try: seen_ids.add(int(tid))
                except: pass
    return seen_slugs, seen_ids

# ---------- Tier logic (auto-detect blob) ----------
def assign_tiers(rows: List[Dict]) -> None:
    """
    T1: rank 1-5
    T2: 6-15
    T3: 16-42
    T5: rank == first rank where rating == 1000 (the blob)
    T4: rank > that blob rank (sub-1000 crowd thereafter)
    Fallbacks keep behavior sensible if blob can't be detected.
    """
    def norm_rank(v):
        try: return int(v)
        except: return 10**9
    def norm_rating(v):
        try: return float(v)
        except: return None

    # sort by rank so first_1000_rank is meaningful
    rows.sort(key=lambda r: norm_rank(r.get("rank")))

    # find first rank with rating == 1000
    first_1000_rank: Optional[int] = None
    for r in rows:
        rating = norm_rating(r.get("rating"))
        if rating is not None and abs(rating - 1000.0) < 1e-6:
            first_1000_rank = norm_rank(r.get("rank"))
            break

    for r in rows:
        rk = norm_rank(r.get("rank"))
        rating = norm_rating(r.get("rating"))

        if 1 <= rk <= 5:
            tier = "T1"
        elif 6 <= rk <= 15:
            tier = "T2"
        elif 16 <= rk <= 42:
            tier = "T3"
        else:
            if first_1000_rank is not None:
                if rk == first_1000_rank:
                    tier = "T5"          # the blob rank (shared)
                elif rk > first_1000_rank:
                    tier = "T4"          # everything after the blob (these are <1000)
                else:
                    # rk < 16 handled above; rk=43..(blob-1) rare; fall back by rating
                    if rating is not None and abs(rating - 1000.0) < 1e-6:
                        tier = "T5"
                    elif rating is not None and rating < 1000.0:
                        tier = "T4"
                    else:
                        tier = "T5"
            else:
                # If we couldn't detect the blob, infer by rating
                if rating is not None and abs(rating - 1000.0) < 1e-6:
                    tier = "T5"
                elif rating is not None and rating < 1000.0:
                    tier = "T4"
                else:
                    tier = "T5"

        r["tier"] = tier
        r["first_1000_rank"] = first_1000_rank if first_1000_rank is not None else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="run headless (default: headed)")
    ap.add_argument("--limit", type=int, default=99999999, help="max teams to collect THIS RUN")
    ap.add_argument("--start-page", type=int, default=1, help="start crawling at this page number")
    ap.add_argument("--max-pages", type=int, default=999999, help="hard cap on how many pages to attempt")
    ap.add_argument("--seed", type=str, default="", help="existing CSV to skip already-seen teams")
    ap.add_argument("--append", action="store_true", help="force append new rows to --out (header kept if file empty)")
    ap.add_argument("--out", type=str, default="gosu_dota2_rankings.csv", help="CSV path (default: gosu_dota2_rankings.csv)")
    args = ap.parse_args()

    log(f"[start] headed={not args.headless} limit={args.limit} start={args.start_page} max_pages={args.max_pages} seed={bool(args.seed)} out={args.out}")

    seed_slugs, seed_ids = load_seed_csv(args.seed)
    log(f"[seed] slugs={len(seed_slugs)} ids={len(seed_ids)}")

    rows, seen_slugs, seen_ids = [], set(seed_slugs), set(seed_ids)

    with sync_playwright() as p:
        log("[playwright] launching chromium")
        browser = p.chromium.launch(
            headless=args.headless,
            slow_mo=70 if not args.headless else 0,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(
            viewport={"width": 1400, "height": 900},
            locale="en-US",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
        )
        # mask webdriver
        ctx.add_init_script("""() => { Object.defineProperty(navigator, 'webdriver', {get: () => undefined}); }""")

        page = ctx.new_page()
        log(f"[nav] {RANKINGS_URL}")
        page.goto(RANKINGS_URL, wait_until="domcontentloaded")
        click_cookie_banners(page)

        if not wait_for_rankings(page):
            log("[error] rankings never appeared; aborting.")
            ctx.close(); browser.close(); sys.exit(2)

        jiggle(page)

        # attempt direct jump to start page; else step via chevron/load-more
        target = max(1, args.start_page)
        last_seen = get_last_page(page)
        log(f"[pager] last_seen={last_seen} target={target}")

        if target > 1 and not click_page(page, target):
            log(f"[pager] direct to page {target} failed; stepping…")
            steps = 0
            while get_current_page(page) < target and steps < (target + 200):
                if not click_next_chevron(page):
                    if not click_load_more(page):
                        break
                steps += 1
                jiggle(page)

        page_num = get_current_page(page)
        log(f"[pager] starting at page {page_num}")

        stalled = 0
        collected = 0
        pages_visited = 0

        while collected < args.limit and stalled < 3 and pages_visited < args.max_pages:
            pages_visited += 1
            jiggle(page)
            wait_for_rankings(page)

            anchors = page.locator(ROW_SEL)
            count = anchors.count()
            log(f"[page {page_num}] anchors={count} collected={collected}")

            before = collected
            for i in range(count):
                a = anchors.nth(i)
                href = a.get_attribute("href") or ""
                if "/dota2/teams/" not in href:
                    continue
                slug = parse_slug(href)
                if not slug or slug in seen_slugs:
                    continue

                name = get_team_name(a)
                team_id = parse_team_id(slug)
                if team_id and team_id in seen_ids:
                    continue

                rank = get_rank_near(a)
                rating = get_rating_near(a)

                rows.append({
                    "team": name,
                    "team_id": team_id,
                    "rank": rank,
                    "rating": rating,
                    "slug": slug,
                    "url": f"{BASE}/dota2/teams/{slug}",
                })
                seen_slugs.add(slug)
                if team_id: seen_ids.add(team_id)
                collected += 1
                if collected >= args.limit:
                    break

            if collected == before:
                stalled += 1
                log(f"[page {page_num}] no new rows (stall {stalled}/3)")
            else:
                stalled = 0

            if collected >= args.limit:
                break

            # move forward aggressively
            moved = False
            cur = get_current_page(page)
            last_now = get_last_page(page)
            if last_now and cur < last_now and click_page(page, cur + 1):
                page_num = cur + 1; moved = True
            elif click_next_chevron(page):
                page_num = get_current_page(page) or (cur + 1); moved = True
            elif click_load_more(page):
                moved = True  # same page expands
            else:
                log("[pager] no next page/loader; stopping.")
                break

        ctx.close(); browser.close()

    # Assign tiers (auto-detect blob rank)
    assign_tiers(rows)

    # CSV Output
    fieldnames = ["team", "team_id", "rank", "rating", "tier", "first_1000_rank", "slug", "url"]

    append_mode = args.append and os.path.exists(args.out)
    if append_mode:
        with open(args.out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if os.path.getsize(args.out) == 0:
                w.writeheader()
            for r in rows:
                w.writerow(r)
        log(f"[done] appended {len(rows)} rows to {args.out}")
    else:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        log(f"[done] wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
