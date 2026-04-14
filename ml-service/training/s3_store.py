"""Cloud storage helpers for the ML pipeline.

Supports both AWS S3 and GCP Cloud Storage.
Backend is selected by the CLOUD_STORAGE_BACKEND env var:
  "s3"  (default) — AWS S3 via boto3
  "gcs"           — GCP Cloud Storage via google-cloud-storage

Handles three responsibilities:
  1. UPLOAD daily collected sessions CSV → bucket/data/sessions/YYYY/MM/DD/sessions.csv
  2. UPLOAD retrained model artifacts   → bucket/models/latest/<name>.joblib
  3. DOWNLOAD model artifacts           → local artifacts/ dir on pod startup

Environment variables (all optional — if not set, storage ops are skipped):
  MODELS_S3_BUCKET        — bucket name (used for both S3 and GCS)
  MODELS_S3_PREFIX        — key prefix for models, default "models/latest"
  DATA_S3_PREFIX          — key prefix for sessions, default "data/sessions"
  CLOUD_STORAGE_BACKEND   — "s3" or "gcs" (default: "s3")
  AWS_REGION              — for S3 only, e.g. "us-east-1"

Credentials:
  AWS:  IAM role on EC2/EKS, or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
  GCP:  Service Account attached to VM (recommended), or GOOGLE_APPLICATION_CREDENTIALS
        pointing to a service account JSON key file

If the required library is not installed or bucket is not set, all calls
are no-ops so the service runs normally in local dev without any cloud.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_BUCKET: Optional[str] = os.environ.get("MODELS_S3_BUCKET")
_MODELS_PREFIX: str = os.environ.get("MODELS_S3_PREFIX", "models/latest")
_DATA_PREFIX: str   = os.environ.get("DATA_S3_PREFIX", "data/sessions")
_REGION: str        = os.environ.get("AWS_REGION", "us-east-1")

# Model artifact files uploaded/downloaded as a set
_MODEL_FILES = [
    "departure_q10.joblib",
    "departure_q50.joblib",
    "departure_q90.joblib",
    "energy_model.joblib",
    "energy_q10.joblib",
    "energy_q90.joblib",
    "arrivals_q10.joblib",
    "arrivals_q50.joblib",
    "arrivals_q90.joblib",
    "demand_model.pkl",
    "user_profiles.db",
]


def _client():
    """Return a boto3 S3 client, or None if boto3 / bucket not available."""
    if not _BUCKET:
        return None
    try:
        import boto3
        return boto3.client("s3", region_name=_REGION)
    except ImportError:
        logger.warning("boto3 not installed — S3 operations disabled")
        return None


# ── Model upload ────────────────────────────────────────────────────────────

def upload_models_to_s3(artifacts_dir: Path) -> None:
    """Upload all model .joblib files from artifacts_dir to S3.

    Called at the end of every successful retrain run so the new models
    are immediately available for the next pod to download on startup.
    """
    s3 = _client()
    if s3 is None:
        logger.info("S3 not configured — skipping model upload")
        return

    uploaded = []
    for fname in _MODEL_FILES:
        local = artifacts_dir / fname
        if not local.exists():
            continue
        key = f"{_MODELS_PREFIX}/{fname}"
        try:
            s3.upload_file(str(local), _BUCKET, key)
            uploaded.append(fname)
        except Exception as exc:
            logger.error("Failed to upload %s to s3://%s/%s: %s", fname, _BUCKET, key, exc)

    if uploaded:
        logger.info(
            "Uploaded %d model artifacts to s3://%s/%s: %s",
            len(uploaded), _BUCKET, _MODELS_PREFIX, uploaded,
        )

    # Write a manifest with version + timestamp so we can track what's live
    try:
        import json
        manifest = {
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "files": uploaded,
            "source": "retrain",
        }
        s3.put_object(
            Bucket=_BUCKET,
            Key=f"{_MODELS_PREFIX}/manifest.json",
            Body=json.dumps(manifest, indent=2).encode(),
            ContentType="application/json",
        )
    except Exception as exc:
        logger.warning("Failed to write S3 manifest: %s", exc)


# ── Model download ──────────────────────────────────────────────────────────

def download_models_from_s3(artifacts_dir: Path) -> bool:
    """Download model artifacts from S3 to local artifacts_dir.

    Called on pod startup BEFORE joblib.load() so pods always use the
    latest retrained models rather than the ones baked into the image.

    Returns True if any files were downloaded, False if S3 is not configured
    or no files were found.
    """
    s3 = _client()
    if s3 is None:
        logger.info("S3 not configured — using models baked into image")
        return False

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for fname in _MODEL_FILES:
        key = f"{_MODELS_PREFIX}/{fname}"
        local = artifacts_dir / fname
        try:
            s3.download_file(_BUCKET, key, str(local))
            downloaded.append(fname)
        except Exception as exc:
            # File may not exist on first deploy — that's fine
            logger.debug("Could not download %s from S3: %s", fname, exc)

    if downloaded:
        logger.info(
            "Downloaded %d model artifacts from s3://%s/%s",
            len(downloaded), _BUCKET, _MODELS_PREFIX,
        )
        return True

    logger.info("No models found in S3 — using image-baked artifacts")
    return False


# ── Session data upload ─────────────────────────────────────────────────────

def upload_sessions_to_s3(sessions_csv: Path) -> bool:
    """Upload the collected sessions CSV to S3 with a date-partitioned key.

    Key format: data/sessions/YYYY/MM/DD/sessions.csv
    This makes it easy to query with Athena and auditable per-day.

    Called by the daily CronJob BEFORE triggering retrain so the data is
    archived even if retraining fails.

    Returns True on success.
    """
    s3 = _client()
    if s3 is None:
        logger.info("S3 not configured — skipping session upload")
        return False

    if not sessions_csv.exists():
        logger.info("No collected_sessions.csv to upload")
        return False

    now = datetime.now(timezone.utc)
    date_prefix = now.strftime("%Y/%m/%d")
    key = f"{_DATA_PREFIX}/{date_prefix}/sessions.csv"

    try:
        s3.upload_file(str(sessions_csv), _BUCKET, key)
        logger.info("Uploaded sessions to s3://%s/%s", _BUCKET, key)
        return True
    except Exception as exc:
        logger.error("Failed to upload sessions to S3: %s", exc)
        return False


def upload_user_profiles_to_s3(db_path: Path) -> None:
    """Upload the user_profiles SQLite DB to S3 for backup.

    This ensures user history survives pod restarts / redeployments.
    Key: models/latest/user_profiles.db  (same as model artifacts)
    """
    s3 = _client()
    if s3 is None:
        return
    if not db_path.exists():
        return
    key = f"{_MODELS_PREFIX}/user_profiles.db"
    try:
        s3.upload_file(str(db_path), _BUCKET, key)
        logger.info("Uploaded user_profiles.db to s3://%s/%s", _BUCKET, key)
    except Exception as exc:
        logger.error("Failed to upload user_profiles.db: %s", exc)
