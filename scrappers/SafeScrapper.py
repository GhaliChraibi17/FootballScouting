#!/usr/bin/env python
"""
Read url_players_belgian_league_2025.csv (col “player_url”),
convert each *profile* URL to its scouting-report URL, and scrape
players with ≥10 s between request starts.
"""

import csv, time, shelve, pathlib
from datetime import datetime
from urllib.parse import urlparse, urlunparse

import pandas as pd
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type
import requests

from scrappers.fbref_scraper import scrape_player

# ——— config
LIST_CSV          = "url_players_netherlands_league_2025.csv"
OUT_CSV           = "Eredivisie_2024_25.csv"
CHECKPOINT_DB     = "scrape_checkpoint.db"
WAIT_SECONDS      = 6.1
SCOUT_CONST       = "12586"        # same for all players / season

HEADERS = {"User-Agent":
   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# ——— helpers
def profile_to_scout(url: str) -> str:
    """
    https://fbref.com/en/players/<pid>/<slug>
      → https://fbref.com/en/players/<pid>/scout/12507/<slug>-Scouting-Report
    """
    parts = url.split("/")
    pid, slug = parts[-2], parts[-1]
    return f"https://fbref.com/en/players/{pid}/scout/{SCOUT_CONST}/{slug}-Scouting-Report"

def load_profiles() -> list[str]:
    df = pd.read_csv(LIST_CSV)
    if "player_url" not in df.columns:
        raise ValueError("CSV must have 'player_url' column")
    return df["player_url"].dropna().tolist()

def already_done(db) -> set[str]:
    if not pathlib.Path(db).exists():
        return set()
    with shelve.open(db) as s:
        return set(s.keys())

def mark_done(db, url):
    with shelve.open(db) as s:
        s[url] = datetime.utcnow().isoformat(timespec="seconds")

# ——— strict-rate wrapper around scrape_player
class TooManyRequests(Exception): ...

@retry(wait=wait_fixed(10), stop=stop_after_attempt(4),
       retry=retry_if_exception_type(TooManyRequests), reraise=True)
def safe_scrape(url):
    try:
        return scrape_player(url)
    except requests.HTTPError as e:
        if e.response.status_code == 429:
            raise TooManyRequests from e
        raise

# ——— main loop
def main():
    profiles = load_profiles()
    done = already_done(CHECKPOINT_DB)
    todo = [u for u in profiles if u not in done]
    print(f"{len(todo)} of {len(profiles)} remaining")

    csv_exists = pathlib.Path(OUT_CSV).exists()
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = None
        last_start = 0.0

        for i, purl in enumerate(todo, 1):
            # hard wait
            dt = time.perf_counter() - last_start
            if dt < WAIT_SECONDS:
                time.sleep(WAIT_SECONDS - dt)
            last_start = time.perf_counter()

            scout_url = profile_to_scout(purl)
            try:
                df = safe_scrape(scout_url)
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=df.columns)
                    if not csv_exists:
                        writer.writeheader()
                writer.writerow(df.iloc[0].to_dict())
                fh.flush()
                mark_done(CHECKPOINT_DB, purl)
                print(f"[{i}/{len(todo)}] ✓ {df.at[0,'Player']}")
            except Exception as e:
                print(f"[{i}/{len(todo)}] ✗ {purl}\n  {e}")

    print("Done, rows in", OUT_CSV)

if __name__ == "__main__":
    main()
