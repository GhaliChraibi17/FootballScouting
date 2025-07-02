import pandas as pd
from pathlib import Path
import re

# ------------------------------------------------------------------
DATA_DIR      = Path("players_data")          # where your CSVs live
OUT_DIR       = Path("players_data_clean")    # cleaned files here
MIN_MINUTES   = 500                           # cut-off
NA_DROP_HOW   = "any"                         # drop rows with *any* NaN
# ------------------------------------------------------------------

def league_from_filename(fname: str) -> str:
    """
    'BelgianPro_2024_25.csv'  -->  'BelgianPro'
    """
    stem = Path(fname).stem                 # drop '.csv'
    parts = stem.rsplit("_", 2)             # split from the right, at most twice
    return parts[0]                         # keep everything before the season pair

def insert_third(df: pd.DataFrame, col_name: str, value):
    """
    Insert column at position 2 (0-indexed).
    """
    df.insert(2, col_name, value)
    return df

def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the first appearance of each column label,
    drop any later duplicates.  Works on pandas ≥1.0.
    """
    return df.loc[:, ~df.columns.duplicated()]

# ---------------------- main loop ---------------------------------
OUT_DIR.mkdir(exist_ok=True)
for csv_path in DATA_DIR.glob("*.csv"):
    print(f"→ processing {csv_path.name}")

    df = pd.read_csv(csv_path)

    # 1️⃣  add League column in 3rd position
    league = league_from_filename(csv_path.name)
    df = insert_third(df, "League", league)

    # 2️⃣  row filtering
    df = df.dropna(how=NA_DROP_HOW)
    if "Minutes" in df.columns:
        df = df[df["Minutes"] >= MIN_MINUTES]

    # 3️⃣  deduplicate column names
    df = drop_duplicate_columns(df)

    # write back
    out_path = OUT_DIR / csv_path.name.replace(".csv", "_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"   saved → {out_path.name}  ({len(df)} rows, {df.shape[1]} cols)")
