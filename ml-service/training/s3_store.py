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
_BACKEND: str       = os.environ.get("CLOUD_STORAGE_BACKEND", "s3").lower()  # "s3" or "gcs"

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


def _s3_client():
    """Return a boto3 S3 client, or None if boto3 / bucket not configured."""
    if not _BUCKET:
        return None
    try:
        import boto3
        return boto3.client("s3", region_name=_REGION)
    except ImportError:
        logger.warning("boto3 not installed — S3 operations disabled")
        return None


def _gcs_bucket():
    """Return a GCS Bucket object, or None if library / bucket not configured."""
    if not _BUCKET:
        return None
    try:
        from google.cloud import storage  # type: ignore
        return storage.Client().bucket(_BUCKET)
    except ImportError:
        logger.warning("google-cloud-storage not installed — GCS operations disabled")
        return None
    except Exception as exc:
        logger.warning("GCS init failed: %s", exc)
        return None


# ── Model upload ────────────────────────────────────────────────────────────

def upload_models_to_s3(artifacts_dir: Path) -> None:
    """Upload all model artifact files from artifacts_dir to cloud storage.

    Called at the end of every successful retrain run so the new models
    are immediately available for the next pod to download on startup.
    Supports both S3 (CLOUD_STORAGE_BACKEND=s3) and GCS (=gcs).
    """
    import json
    uploaded = []

    if _BACKEND == "gcs":
        bucket = _gcs_bucket()
        if bucket is None:
            logger.info("GCS not configured — skipping model upload")
            return
        for fname in _MODEL_FILES:
            local = artifacts_dir / fname
            if not local.exists():
                continue
            key = f"{_MODELS_PREFIX}/{fname}"
            try:
                bucket.blob(key).upload_from_filename(str(local))
                uploaded.append(fname)
            except Exception as exc:
                logger.error("Failed to upload %s to gs://%s/%s: %s", fname, _BUCKET, key, exc)
        if uploaded:
            logger.info("Uploaded %d model artifacts to gs://%s/%s: %s",
                        len(uploaded), _BUCKET, _MODELS_PREFIX, uploaded)
        # Write manifest
        try:
            manifest = {"uploaded_at": datetime.now(timezone.utc).isoformat(),
                        "files": uploaded, "source": "retrain"}
            bucket.blob(f"{_MODELS_PREFIX}/manifest.json").upload_from_string(
                json.dumps(manifest, indent=2).encode(), content_type="application/json")
        except Exception as exc:
            logger.warning("Failed to write GCS manifest: %s", exc)

    else:  # s3
        s3 = _s3_client()
        if s3 is None:
            logger.info("S3 not configured — skipping model upload")
            return
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
            logger.info("Uploaded %d model artifacts to s3://%s/%s: %s",
                        len(uploaded), _BUCKET, _MODELS_PREFIX, uploaded)
        # Write manifest
        try:
            manifest = {"uploaded_at": datetime.now(timezone.utc).isoformat(),
                        "files": uploaded, "source": "retrain"}
            s3.put_object(Bucket=_BUCKET, Key=f"{_MODELS_PREFIX}/manifest.json",
                          Body=json.dumps(manifest, indent=2).encode(),
                          ContentType="application/json")
        except Exception as exc:
            logger.warning("Failed to write S3 manifest: %s", exc)


# ── Model download ──────────────────────────────────────────────────────────

def download_models_from_s3(artifacts_dir: Path) -> bool:
    """Download model artifacts from cloud storage to local artifacts_dir.

    Called on pod startup BEFORE joblib.load() so pods always use the
    latest retrained models rather than the ones baked into the image.
    Supports both S3 (CLOUD_STORAGE_BACKEND=s3) and GCS (=gcs).

    Returns True if any files were downloaded, False otherwise.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    if _BACKEND == "gcs":
        bucket = _gcs_bucket()
        if bucket is None:
            logger.info("GCS not configured — using models baked into image")
            return False
        for fname in _MODEL_FILES:
            key = f"{_MODELS_PREFIX}/{fname}"
            local = artifacts_dir / fname
            try:
                bucket.blob(key).download_to_filename(str(local))
                downloaded.append(fname)
            except Exception as exc:
                logger.debug("Could not download %s from GCS: %s", fname, exc)
        if downloaded:
            logger.info("Downloaded %d model artifacts from gs://%s/%s",
                        len(downloaded), _BUCKET, _MODELS_PREFIX)
            return True
        logger.info("No models found in GCS — using image-baked artifacts")
        return False

    else:  # s3
        s3 = _s3_client()
        if s3 is None:
            logger.info("S3 not configured — using models baked into image")
            return False
        for fname in _MODEL_FILES:
            key = f"{_MODELS_PREFIX}/{fname}"
            local = artifacts_dir / fname
            try:
                s3.download_file(_BUCKET, key, str(local))
                downloaded.append(fname)
            except Exception as exc:
                logger.debug("Could not download %s from S3: %s", fname, exc)
        if downloaded:
            logger.info("Downloaded %d model artifacts from s3://%s/%s",
                        len(downloaded), _BUCKET, _MODELS_PREFIX)
            return True
        logger.info("No models found in S3 — using image-baked artifacts")
        return False


# ── Session data upload ─────────────────────────────────────────────────────

def upload_sessions_to_s3(sessions_csv: Path) -> bool:
    """Upload the collected sessions CSV to cloud storage with a date-partitioned key.

    Key format: data/sessions/YYYY/MM/DD/sessions.csv
    Called by the daily CronJob BEFORE triggering retrain so the data is
    archived even if retraining fails.
    Supports both S3 (CLOUD_STORAGE_BACKEND=s3) and GCS (=gcs).

    Returns True on success.
    """
    if not sessions_csv.exists():
        logger.info("No collected_sessions.csv to upload")
        return False

    now = datetime.now(timezone.utc)
    key = f"{_DATA_PREFIX}/{now.strftime('%Y/%m/%d')}/sessions.csv"

    if _BACKEND == "gcs":
        bucket = _gcs_bucket()
        if bucket is None:
            logger.info("GCS not configured — skipping session upload")
            return False
        try:
            bucket.blob(key).upload_from_filename(str(sessions_csv))
            logger.info("Uploaded sessions to gs://%s/%s", _BUCKET, key)
            return True
        except Exception as exc:
            logger.error("Failed to upload sessions to GCS: %s", exc)
            return False

    else:  # s3
        s3 = _s3_client()
        if s3 is None:
            logger.info("S3 not configured — skipping session upload")
            return False
        try:
            s3.upload_file(str(sessions_csv), _BUCKET, key)
            logger.info("Uploaded sessions to s3://%s/%s", _BUCKET, key)
            return True
        except Exception as exc:
            logger.error("Failed to upload sessions to S3: %s", exc)
            return False


def upload_user_profiles_to_s3(db_path: Path) -> None:
    """Upload the user_profiles SQLite DB to cloud storage for backup.

    This ensures user history survives pod restarts / redeployments.
    Key: models/latest/user_profiles.db
    Supports both S3 (CLOUD_STORAGE_BACKEND=s3) and GCS (=gcs).
    """
    if not db_path.exists():
        return
    key = f"{_MODELS_PREFIX}/user_profiles.db"

    if _BACKEND == "gcs":
        bucket = _gcs_bucket()
        if bucket is None:
            return
        try:
            bucket.blob(key).upload_from_filename(str(db_path))
            logger.info("Uploaded user_profiles.db to gs://%s/%s", _BUCKET, key)
        except Exception as exc:
            logger.error("Failed to upload user_profiles.db to GCS: %s", exc)

    else:  # s3
        s3 = _s3_client()
        if s3 is None:
            return
        try:
            s3.upload_file(str(db_path), _BUCKET, key)
            logger.info("Uploaded user_profiles.db to s3://%s/%s", _BUCKET, key)
        except Exception as exc:
            logger.error("Failed to upload user_profiles.db to S3: %s", exc)
