#!/usr/bin/env python
# fbref_scraper.py  ─────────────────────────────────────────────
"""
scrape_player(url, *, to_csv=None)  →  tidy 1×N DataFrame

Columns (first six):
    Player, Birthdate, Club, Footed, Nationality, Position, Minutes, …
"""

from __future__ import annotations
import re, warnings
from io import StringIO
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import sys
import requests
from bs4 import BeautifulSoup

# ————————————————— HTTP
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

def _download(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text

def _strip_comments(html: str) -> str:
    return html.replace("<!--", "").replace("-->", "")


# ————————————————— scout_full_* table  (id starts with scout_full_)
def _get_scout_full_table(soup: BeautifulSoup) -> pd.DataFrame:
    tbl = soup.select_one('table[id^="scout_full_"]')
    if not tbl:
        raise ValueError("No table with id starting “scout_full_” found.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        df = pd.read_html(StringIO(str(tbl)), flavor="lxml")[0]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(c).strip("_") for c in df.columns]
    return df


# ————————————————— meta extractor (unchanged except for return slot order)
def _first(it):
    for el in it:
        return el
    return None

def _collect_meta(soup: BeautifulSoup):
    meta = soup.find(id="meta")
    txt  = meta.get_text(" ", strip=True).replace("\xa0", " ") if meta else ""
    full = soup.get_text(" ", strip=True).replace("\xa0", " ")

    birth = soup.select_one("span#necro-birth[data-birth]")
    birthdate = birth["data-birth"] if birth else None

    club = None
    if meta:
        link = _first(meta.select('a[href^="/en/squads/"]'))
        if link:
            club = link.get_text(strip=True)
    if not club:
        m = re.search(r"Club:\s*([A-Za-z0-9 .'-]+)", txt)
        if m:
            club = m.group(1).strip()

    m = re.search(r"Footed:\s*([A-Za-z]+)", txt, re.I)
    footed = m.group(1).title() if m else None

    nat = None
    if meta:
        link = _first(meta.select('a[href^="/en/country/"]'))
        if link:
            nat = link.get_text(strip=True)
    if not nat:
        m = re.search(r"Nationality:\s*([A-Za-z ]+)", txt)
        if m:
            nat = m.group(1).strip()

    # 1) any footer id that begins with tfooter_scout_summary_
    foot_strong = soup.select_one('div[id^="tfooter_scout_summary_"] strong')
    if foot_strong:
        txt = foot_strong.get_text(" ", strip=True).replace("\xa0", " ")
        m = re.search(r"([\d,]+)\s+minutes", txt, flags=re.I)
        if m:
            minutes = int(m.group(1).replace(",", ""))

    # 2) fallback – first "NN minutes" anywhere on the page
    if minutes is None:
        txt_full = soup.get_text(" ", strip=True).replace("\xa0", " ")
        m = re.search(r"([\d,]+)\s+minutes", txt_full, flags=re.I)
        if m:
            minutes = int(m.group(1).replace(",", ""))

    pos = None
    sw = soup.select_one(".filter.switcher .current a.sr_preset")
    if sw:
        m = re.search(r"vs\.\s*(.+)", sw.get_text(strip=True))
        if m:
            pos = m.group(1).strip()
    if pos is None:
        m = re.search(r"Position[s]?:\s*(.+?)(?:\||$)", txt)
        if m:
            pos = m.group(1).strip()

    return birthdate, club, footed, nat, minutes, pos


# ————————————————— pivot helper
def _pivot_per90(df: pd.DataFrame, *,
                 player: str, birthdate: Optional[str], club: Optional[str],
                 footed: Optional[str], nat: Optional[str],
                 pos: Optional[str], mins: Optional[int]) -> pd.DataFrame:

    vals: Dict[str, float] = {}
    for s in [c for c in df.columns
              if c.lower().endswith("_statistic") or c.lower() == "statistic"]:
        pcol = (s[:-9] + "Per 90") if s.lower().endswith("_statistic") else "Per 90"
        if pcol not in df.columns:
            continue
        for stat, per90 in df[[s, pcol]].dropna(how="all").itertuples(index=False):
            v = pd.to_numeric(str(per90).replace("%", ""), errors="coerce")
            if pd.notna(v):
                vals.setdefault(stat, float(v))

    wide = pd.DataFrame([vals])
    wide.insert(0, "Minutes", mins)
    wide.insert(0, "Position", pos)
    wide.insert(0, "Nationality", nat)
    wide.insert(0, "Footed", footed)
    wide.insert(0, "Club", club)
    wide.insert(0, "Birthdate", birthdate)
    wide.insert(0, "Player", player)
    return wide


# ————————————————— public
def scrape_player(url: str, *, to_csv: str | None = None) -> pd.DataFrame:
    html = _download(url)
    soup = BeautifulSoup(_strip_comments(html), "lxml")

    df_full = _get_scout_full_table(soup)
    birth, club, foot, nat, mins, pos = _collect_meta(soup)
    player  = soup.find("h1").get_text(strip=True)

    row = _pivot_per90(df_full, player=player, birthdate=birth, club=club,
                       footed=foot, nat=nat, pos=pos, mins=mins)

    if to_csv:
        Path(to_csv).parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(to_csv, index=False)
    return row


# ————————————————— CLI
if __name__ == "__main__" and len(sys.argv) > 1:
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("url"), pa.add_argument("--out")
    a = pa.parse_args()
    print(scrape_player(a.url, to_csv=a.out).iloc[:, :15])
