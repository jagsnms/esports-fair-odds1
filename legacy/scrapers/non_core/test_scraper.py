import re
import asyncio
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

BASE = "https://liquipedia.net"
URL  = f"{BASE}/dota2/Liquipedia:Matches?status=completed"

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower().replace("_", " ")).replace("team ", "").strip()

def _grab_team_name(opponent_el) -> str:
    """
    Team name is often NOT in text; prefer attributes.
    Priority: .team-template-text a[title] -> a[data-team] -> img[alt] -> visible text.
    """
    # 1) Typical case: link with title attribute
    a = opponent_el.select_one(".team-template-text a")
    if a:
        title = a.get("title")
        if title: return title.strip()
        dt = a.get("data-team")
        if dt: return dt.strip()
        if a.text.strip(): return a.text.strip()

    # 2) Sometimes they use a different anchor
    a2 = opponent_el.select_one(".block-team a")
    if a2:
        title = a2.get("title")
        if title: return title.strip()
        dt = a2.get("data-team")
        if dt: return dt.strip()
        if a2.text.strip(): return a2.text.strip()

    # 3) Fallback: logo alt text
    img = opponent_el.select_one("img[alt]")
    if img and img.get("alt"): 
        return img["alt"].strip()

    # 4) Last resort: any visible text
    return opponent_el.get_text(" ", strip=True)

def _grab_row_score(header_el):
    """
    Try to read series score off the list row.
    Returns (left_score, right_score) or (None, None) if not present.
    """
    sh = header_el.select_one(".match-info-header-scoreholder")
    if not sh:
        return (None, None)

    def score_from(box_sel: str):
        box = sh.select_one(box_sel)
        if not box:
            return None
        s = box.select_one(".match-info-header-score")
        if s:
            txt = s.get_text(strip=True)
            if txt.isdigit():
                return int(txt)

        # Some builds hide text but place numbers in attributes
        for attr in ("data-score", "aria-label", "title"):
            val = box.get(attr)
            if val:
                m = re.search(r"\d+", val)
                if m: 
                    try: return int(m.group())
                    except: pass
        return None

    left = score_from(".match-info-header-scoreholder-upper")
    right = score_from(".match-info-header-scoreholder-lower")
    return (left, right)

def _extract_row_link(header_el):
    # Main clickable link to the match detail
    a = header_el.select_one("a.match-info-link, a.match-info")
    if a and a.get("href"):
        href = a["href"]
        return href if href.startswith("http") else urljoin(BASE, href)
    # Fallback: any anchor under header
    a2 = header_el.select_one("a[href]")
    if a2 and a2.get("href"):
        href = a2["href"]
        return href if href.startswith("http") else urljoin(BASE, href)
    return None

def _parse_detail_for_score(html: str):
    """
    On match detail page, find a '2-1' or '1–0' etc anywhere reasonable.
    Returns (left_score, right_score) or (None, None).
    """
    soup = BeautifulSoup(html, "lxml")

    # Common containers to scan first
    candidates = []
    candidates += [el.get_text(" ", strip=True) for el in soup.select(".match-info, .brkts-match, .series, .match, #bodyContent, #mw-content-text")]

    # Deduplicate long strings a bit
    seen = set()
    texts = []
    for t in candidates:
        if t and t not in seen:
            texts.append(t)
            seen.add(t)

    # Look for something like 2-1 / 1–0 / 0 — 2, etc
    pat = re.compile(r"\b(\d+)\s*[–\-—]\s*(\d+)\b")
    for t in texts:
        m = pat.search(t)
        if m:
            try:
                l = int(m.group(1))
                r = int(m.group(2))
                return (l, r)
            except:
                continue

    # Some pages have explicit score nodes
    l = soup.select_one(".match-score .team-left .score, .team-left .match-score")
    r = soup.select_one(".match-score .team-right .score, .team-right .match-score")
    try_l = int(l.get_text(strip=True)) if l and l.get_text(strip=True).isdigit() else None
    try_r = int(r.get_text(strip=True)) if r and r.get_text(strip=True).isdigit() else None
    if try_l is not None and try_r is not None:
        return (try_l, try_r)

    return (None, None)

async def _render_completed_with_scroll(min_rows=80, max_scrolls=12, wait_ms=700):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent="Mozilla/5.0")
        page = await ctx.new_page()
        await page.goto(URL, wait_until="domcontentloaded", timeout=30000)
        try:
            await page.wait_for_selector(".match-info-header", timeout=8000)
        except PWTimeout:
            pass

        last_count = 0
        for _ in range(max_scrolls):
            count = await page.locator(".match-info-header").count()
            if count >= min_rows and count == last_count:
                break
            last_count = count
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(wait_ms)

        html = await page.content()
        await browser.close()
        return html

async def _fetch_detail_html(url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent="Mozilla/5.0")
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        # Try to let dynamic bits settle
        await page.wait_for_timeout(1000)
        html = await page.content()
        await browser.close()
        return html

def _parse_completed_rows_for_team(list_html: str, team_name: str, need: int, debug: bool = False):
    team_norm = _norm(team_name)
    soup = BeautifulSoup(list_html, "lxml")
    out = []

    for hdr in soup.select("div.match-info-header"):
        opps = hdr.select(".match-info-header-opponent")
        if len(opps) != 2:
            continue

        name_left  = _grab_team_name(opps[0])
        name_right = _grab_team_name(opps[1])
        if not name_left or not name_right:
            continue

        left_norm, right_norm = _norm(name_left), _norm(name_right)
        if team_norm not in (left_norm, right_norm):
            continue

        sL, sR = _grab_row_score(hdr)

        link = _extract_row_link(hdr)  # keep for fallback

        out.append({
            "left_name": name_left,
            "right_name": name_right,
            "left_score": sL,
            "right_score": sR,
            "link": link
        })
        if debug:
            print(f"[row] {name_left} [{sL}] vs {name_right} [{sR}] link={bool(link)}")

        if len(out) >= need * 2:   # collect a few extras in case of draws/None
            break

    return out

async def _resolve_scores_if_needed(rows, need: int, debug: bool = False):
    """
    For any row where scores are missing, fetch the detail page and parse the series score.
    Returns list of dicts with 'opponent' and 'win'.
    """
    results = []
    for row in rows:
        nL, nR = row["left_name"], row["right_name"]
        sL, sR = row["left_score"], row["right_score"]

        if sL is None or sR is None:
            if row["link"]:
                try:
                    html = await _fetch_detail_html(row["link"])
                    sL2, sR2 = _parse_detail_for_score(html)
                    if debug:
                        print(f"[detail] {nL} vs {nR} -> {sL2}-{sR2}")
                    sL, sR = sL2, sR2
                except Exception as e:
                    if debug:
                        print(f"[detail error] {e}")

        # Skip if still unknown or if draw
        if sL is None or sR is None or sL == sR:
            continue

        row["left_score"], row["right_score"] = sL, sR
        results.append(row)
        if len([r for r in results if r["left_score"] is not None]) >= need:
            break

    return results

async def _fetch_recent_matches_async(team_name: str, limit: int = 15, debug: bool = False):
    list_html = await _render_completed_with_scroll()
    candidate_rows = _parse_completed_rows_for_team(list_html, team_name, need=limit, debug=debug)
    resolved = await _resolve_scores_if_needed(candidate_rows, need=limit, debug=debug)

    # Build final shape: {opponent, win}
    team_norm = _norm(team_name)
    out = []
    for r in resolved:
        left_norm  = _norm(r["left_name"])
        right_norm = _norm(r["right_name"])
        left_won = r["left_score"] > r["right_score"]
        if team_norm == left_norm:
            out.append({"opponent": r["right_name"], "win": left_won})
        elif team_norm == right_norm:
            out.append({"opponent": r["left_name"], "win": (not left_won)})

        if len(out) >= limit:
            break

    if debug:
        print(f"[lp] final rows={len(out)} for team={team_name}")

    return out

def fetch_recent_matches_liquipedia(team_name: str, limit: int = 15, debug: bool = False):
    """Sync wrapper for Streamlit."""
    return asyncio.run(_fetch_recent_matches_async(team_name, limit=limit, debug=debug))
