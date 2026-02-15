# liquipedia_scrape_playwright.py
# Requires:
#   pip install playwright==1.46.0 beautifulsoup4 lxml
#   python -m playwright install chromium

import re
import asyncio
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

BASE = "https://liquipedia.net"
URL  = f"{BASE}/dota2/Liquipedia:Matches?status=completed"

# -----------------------------
# Helpers
# -----------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower().replace("_", " ")).replace("team ", "").strip()

def _grab_team_name(opponent_el) -> str:
    """
    Team name is often NOT visible as text. Prefer attributes.
    Priority: .team-template-text a[title] -> a[data-team] -> img[alt] -> visible text.
    """
    a = opponent_el.select_one(".team-template-text a")
    if a:
        title = a.get("title")
        if title: return title.strip()
        dt = a.get("data-team")
        if dt: return dt.strip()
        if a.text.strip(): return a.text.strip()

    a2 = opponent_el.select_one(".block-team a")
    if a2:
        title = a2.get("title")
        if title: return title.strip()
        dt = a2.get("data-team")
        if dt: return dt.strip()
        if a2.text.strip(): return a2.text.strip()

    img = opponent_el.select_one("img[alt]")
    if img and img.get("alt"):
        return img["alt"].strip()

    return opponent_el.get_text(" ", strip=True)

def _grab_row_score(header_el) -> Tuple[Optional[int], Optional[int]]:
    """
    Try to read series score off the list row (if rendered).
    Returns (left_score, right_score) or (None, None).
    """
    sh = header_el.select_one(".match-info-header-scoreholder")
    if not sh:
        return (None, None)

    def score_from(box_sel: str):
        box = sh.select_one(box_sel)
        if not box:
            return None

        # Visible number?
        s = box.select_one(".match-info-header-score")
        if s:
            txt = s.get_text(strip=True)
            if txt.isdigit():
                return int(txt)

        # Hidden in attributes?
        for attr in ("data-score", "aria-label", "title"):
            val = box.get(attr)
            if val:
                m = re.search(r"\b([0-5])\b", val)
                if m:
                    try:
                        n = int(m.group(1))
                        if 0 <= n <= 5:
                            return n
                    except:
                        pass
        return None

    left = score_from(".match-info-header-scoreholder-upper")
    right = score_from(".match-info-header-scoreholder-lower")
    return (left, right)

def _extract_row_link(header_el) -> Optional[str]:
    a = header_el.select_one("a.match-info-link, a.match-info")
    if a and a.get("href"):
        href = a["href"]
        return href if href.startswith("http") else urljoin(BASE, href)
    a2 = header_el.select_one("a[href]")
    if a2 and a2.get("href"):
        href = a2["href"]
        return href if href.startswith("http") else urljoin(BASE, href)
    return None

def _parse_detail_for_score(html: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse match detail page for series score.
    Strict: only accept X–Y where X,Y in 0..5 (avoids dates like 2014-10).
    """
    soup = BeautifulSoup(html, "lxml")
    # 1) Obvious score-like elements first
    score_nodes = soup.select(
        ".match-info-header-score, .match-score, .series-score, "
        ".brkts-match .brkts-score, .brkts-match .score, "
        ".series .score, .infobox-match .score"
    )
    for el in score_nodes:
        txt = el.get_text(" ", strip=True)
        m = re.search(r"\b([0-5])\s*[–\-—]\s*([0-5])\b", txt)
        if m:
            return int(m.group(1)), int(m.group(2))

    # 2) Attributes sometimes hold it
    for el in soup.select("[aria-label],[title],[data-score]"):
        for attr in ("aria-label", "title", "data-score"):
            val = el.get(attr)
            if not val:
                continue
            m = re.search(r"\b([0-5])\s*[–\-—]\s*([0-5])\b", val)
            if m:
                return int(m.group(1)), int(m.group(2))

    # 3) Broader text scan within likely containers (still strict 0..5)
    blocks = soup.select(".match-info, .brkts-match, .series, .match, #mw-content-text, #bodyContent")
    seen = set()
    for b in blocks:
        txt = b.get_text(" ", strip=True)
        if not txt or txt in seen:
            continue
        seen.add(txt)
        m = re.search(r"\b([0-5])\s*[–\-—]\s*([0-5])\b", txt)
        if m:
            return int(m.group(1)), int(m.group(2))

    return (None, None)

# -----------------------------
# Playwright flows
# -----------------------------
async def _render_completed_with_scroll(min_rows=80, max_scrolls=14, wait_ms=750) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent="Mozilla/5.0")
        page = await ctx.new_page()

        await page.goto(URL, wait_until="domcontentloaded", timeout=30000)
        try:
            await page.wait_for_selector(".match-info-header", timeout=9000)
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

def _collect_candidate_rows(list_html: str, team_name: str, need: int, debug: bool) -> List[Dict]:
    team_norm = _norm(team_name)
    soup = BeautifulSoup(list_html, "lxml")
    out = []

    for hdr in soup.select("div.match-info-header"):
        opps = hdr.select(".match-info-header-opponent")
        if len(opps) != 2:
            continue

        nL = _grab_team_name(opps[0])
        nR = _grab_team_name(opps[1])
        if not nL or not nR:
            continue

        L, R = _norm(nL), _norm(nR)
        if team_norm not in (L, R):
            continue

        sL, sR = _grab_row_score(hdr)
        link = _extract_row_link(hdr)

        row = {
            "left_name": nL,
            "right_name": nR,
            "left_score": sL,
            "right_score": sR,
            "link": link,
        }
        out.append(row)

        if debug:
            print(f"[row] {nL} [{sL}] vs {nR} [{sR}] link={bool(link)}")

        # keep some extras in case of draws/unknowns
        if len(out) >= need * 2:
            break

    return out

async def _resolve_scores_if_needed(rows: List[Dict], team_name: str, need: int, debug: bool) -> List[Dict]:
    """
    For rows with missing/None scores, open the detail page once and parse the series score.
    """
    if not rows:
        return []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent="Mozilla/5.0")
        page = await ctx.new_page()

        resolved = []
        for row in rows:
            sL, sR = row["left_score"], row["right_score"]
            nL, nR = row["left_name"], row["right_name"]

            if sL is None or sR is None:
                if row["link"]:
                    try:
                        await page.goto(row["link"], wait_until="domcontentloaded", timeout=30000)
                        await page.wait_for_timeout(900)  # let the template settle
                        html = await page.content()
                        sL2, sR2 = _parse_detail_for_score(html)
                        if debug:
                            print(f"[detail] {nL} vs {nR} -> {sL2}-{sR2}")
                        sL, sR = sL2, sR2
                    except Exception as e:
                        if debug:
                            print(f"[detail error] {e}")

            # skip draws or still-unknowns
            if sL is None or sR is None or sL == sR:
                continue

            row["left_score"], row["right_score"] = sL, sR
            resolved.append(row)

            if len(resolved) >= need:
                break

        await browser.close()
        return resolved

async def _fetch_recent_matches_async(team_name: str, limit: int = 15, debug: bool = False) -> List[Dict]:
    list_html = await _render_completed_with_scroll()
    candidates = _collect_candidate_rows(list_html, team_name, need=limit, debug=debug)
    rows = await _resolve_scores_if_needed(candidates, team_name, need=limit, debug=debug)

    # Final mapping to your app format
    team_norm = _norm(team_name)
    out = []
    for r in rows:
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

# -----------------------------
# Public sync wrapper
# -----------------------------
def fetch_recent_matches_liquipedia(team_name: str, limit: int = 15, debug: bool = False) -> List[Dict]:
    """
    Returns: [{"opponent": <str>, "win": <bool>}, ...]
    """
    return asyncio.run(_fetch_recent_matches_async(team_name, limit=limit, debug=debug))


# -----------------------------
# CLI test
# -----------------------------
if __name__ == "__main__":
    import sys
    team = "eSpoiled" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    rows = fetch_recent_matches_liquipedia(team, limit=10, debug=True)
    print(rows)
