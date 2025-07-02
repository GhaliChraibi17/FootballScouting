#!/usr/bin/env python
"""
fc25_rating_scraper.py
======================

Scrapes EA Sports FC 24 Overall rating, Height (cm) and Full Name for
every player listed on SoFIFA in

 • Premier League (lg=13)
 • EFL Championship (lg=14)

Handles Cloudflare (via cloudscraper) and skips the “+1” boost badge by
pulling the rating from <em title="..">.
"""

import csv, random, time
from pathlib import Path
from typing import Iterator, Tuple, List

import cloudscraper
from bs4 import BeautifulSoup

# ───────────── constants
BASE          = "https://sofifa.com"
LEAGUE_IDS    = [13, 14]        # PL, Championship
ROWS_PER_PAGE = 60
WAIT_MIN, WAIT_MAX = 1.2, 1.8
SAVE_DIR      = Path("fc25_dump")
OUT_CSV       = SAVE_DIR / "PL_CH_f24.csv"

SESSION = cloudscraper.create_scraper(
    delay=5,
    browser={"browser": "chrome", "platform": "windows", "mobile": False},
)
SESSION.headers.update({
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://google.com",
    "Connection": "keep-alive",
})

# ───────────── helpers
def fetch(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        r = SESSION.get(url, timeout=30)
        if r.status_code in (403, 429):
            t = 5 + attempt * 5
            print(f"  {r.status_code} on {url} → sleep {t}s")
            time.sleep(t); continue
        r.raise_for_status(); return r.text
    raise RuntimeError(f"failed after {retries} tries → {url}")


def league_rows(lg_ids: List[int]) -> Iterator[Tuple[str, str, str, int]]:
    """
    Yield (player_url, full_name, club_name, overall)
    for all leagues in lg_ids.
    """
    query = "&".join([f"lg%5B%5D={i}" for i in lg_ids])
    offset = 0

    while True:
        url  = f"{BASE}/players?type=all&{query}&offset={offset}"
        soup = BeautifulSoup(fetch(url), "lxml")
        rows = soup.select("table tbody tr")
        if not rows:
            break

        for tr in rows:
            a_player = tr.select_one('a[href^="/player/"]')
            if not a_player:
                continue
            full_name = a_player["data-tippy-content"]
            player_url = BASE + a_player["href"]

            # --- Overall -------------------------------------------
            ovr_td = tr.select_one('td[data-col="oa"]')
            em     = ovr_td.select_one("em")
            try:
                overall = int(em.get("title") or em.text.strip())
            except ValueError:
                continue                                  # malformed row

            # --- Club name -----------------------------------------
            club_tag = tr.select_one('a[href^="/team/"]')
            club_name = club_tag.get_text(strip=True) if club_tag else ""

            yield player_url, full_name, club_name, overall

        offset += ROWS_PER_PAGE
        time.sleep(random.uniform(WAIT_MIN, WAIT_MAX))



# ───────────── main
def main():
    SAVE_DIR.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["player_url", "full_name", "overall", "club_name"])

        for i, row in enumerate(league_rows(LEAGUE_IDS), 1):
            w.writerow(row)
            if i % 100 == 0:
                print(f"  scraped {i} players …")

    print(f"\n✓ saved {OUT_CSV.relative_to(SAVE_DIR.parent)}")


if __name__ == "__main__":
    import sys
    try:
        import bs4, cloudscraper  # ensure deps present
    except ImportError:
        sys.exit("Install deps:  pip install cloudscraper beautifulsoup4 lxml")
    main()
