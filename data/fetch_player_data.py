from pathlib import Path
import time
import pandas as pd

from nba_api.stats.endpoints import (
    leaguegamelog,
    boxscoretraditionalv3,
    boxscoreadvancedv3,
)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

SEASONS = [
    "2024-25",
    "2025-26",
]