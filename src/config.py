"""
Paths, seasons and leagues, shared by all scripts.
Set FOOTBALL_DATA_DIR as an env var to keep data outside the project (e.g. outside OneDrive).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FOOTBALL_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
UNDERSTAT_DIR = RAW_DIR / "understat"
PROCESSED_DIR = DATA_DIR / "processed"

# start year of the season, 2014 = 2014-15. understat's own "season" column uses the same convention.
# the dumps have complete seasons up to 2023-24, 2024-25 only has a few EPL rounds so it's left out
SEASONS = range(2014, 2024)

# big five only. RFPL is in the dumps too but barely shares players with the rest, left out for now
LEAGUES = ["EPL", "La liga", "Bundesliga", "Serie A", "Ligue 1"]

# nominal match length for segment durations. stoppage time is in the shot minutes (91, 92, ...) but
# not in the roster minutes (capped at 90), so the last segment of each match gets this much extra
STOPPAGE = 4


def season_label(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"
