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

# User profile store
# Local: SQLite file  /  AWS: set USER_PROFILES_BACKEND=dynamodb
USER_PROFILES_DB = Path(os.environ.get("USER_PROFILES_DB", str(ARTIFACTS_DIR / "user_profiles.db")))

# S3 — set MODELS_S3_BUCKET to enable; all other S3 vars have defaults
# On AWS EKS, use an IAM role attached to the service account (no key needed).
# For local testing: set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY env vars.
MODELS_S3_BUCKET = os.environ.get("MODELS_S3_BUCKET", "")   # empty = S3 disabled
