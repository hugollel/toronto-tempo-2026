"""
01_ingest.py — Raw data ingestion
=================================
Downloads public WNBA data (ESPN, via the SportsDataverse / wehoop releases)
and stores it untouched in data/raw/. Every download is logged in
data/raw/ingestion_log.csv (what, when, how many rows, success or failure).

Rule of the pipeline: raw data is never modified. Cleaning happens later,
in a separate step, so we can always go back to the original source.

Run from the project folder:
    python3 pipeline/01_ingest.py
"""

from datetime import datetime, timezone
from pathlib import Path
import hashlib

import pandas as pd
import requests

# --- Settings ---------------------------------------------------------------

SEASONS = [2024, 2025, 2026]

BASE_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"

# dataset name -> (release tag, file name pattern)
DATASETS = {
    "schedule":   ("espn_wnba_schedules",        "wnba_schedule_{season}"),
    "team_box":   ("espn_wnba_team_boxscores",   "team_box_{season}"),
    "player_box": ("espn_wnba_player_boxscores", "player_box_{season}"),
    "pbp":        ("espn_wnba_pbp",              "play_by_play_{season}"),
}

FORMATS = ["parquet", "csv"]  # tried in this order

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
LOG_FILE = RAW_DIR / "ingestion_log.csv"


# --- Helpers ----------------------------------------------------------------

def download(url: str, dest: Path) -> bool:
    """Download a file. Returns True if it worked, False otherwise."""
    try:
        r = requests.get(url, timeout=60)
    except requests.RequestException as e:
        print(f"   network error: {e}")
        return False
    if r.status_code != 200:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return True


def read_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def md5(path: Path) -> str:
    """Fingerprint of the file, to detect if the source changed later."""
    return hashlib.md5(path.read_bytes()).hexdigest()


# --- Main -------------------------------------------------------------------

def main() -> None:
    log_rows = []

    for name, (tag, pattern) in DATASETS.items():
        for season in SEASONS:
            stem = pattern.format(season=season)
            status, rows, cols, checksum, used_url = "not_found", None, None, None, None

            for fmt in FORMATS:
                url = f"{BASE_URL}/{tag}/{stem}.{fmt}"
                dest = RAW_DIR / name / f"{stem}.{fmt}"
                if download(url, dest):
                    try:
                        df = read_file(dest)
                        status, rows, cols = "ok", len(df), df.shape[1]
                        checksum, used_url = md5(dest), url
                    except Exception as e:
                        status = f"read_error: {e}"
                    break

            symbol = "OK " if status == "ok" else "-- "
            detail = f"{rows} rows x {cols} cols" if status == "ok" else status
            print(f"{symbol} {name:<11} {season}  {detail}")

            log_rows.append({
                "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "dataset": name,
                "season": season,
                "status": status,
                "rows": rows,
                "columns": cols,
                "md5": checksum,
                "source_url": used_url,
            })

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log = pd.DataFrame(log_rows)
    log.to_csv(LOG_FILE, mode="a", header=not LOG_FILE.exists(), index=False)

    n_ok = (log["status"] == "ok").sum()
    print(f"\n{n_ok}/{len(log)} files downloaded. Log: {LOG_FILE.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
