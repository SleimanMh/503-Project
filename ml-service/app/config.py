"""ML Service configuration."""

import os
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
MODEL_VERSION = "1.0.0"

# Shared data directory (mounted volume — same as gateway writes to)
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
FEEDBACK_CSV = DATA_DIR / "departure_feedback.csv"
COLLECTED_SESSIONS_CSV = DATA_DIR / "collected_sessions.csv"
ACN_DATA_PATH = Path(os.environ.get("ACN_DATA_PATH", "/app/data/acn_sessions.csv"))
