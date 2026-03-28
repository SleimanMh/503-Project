"""Gateway configuration — URLs of internal services."""

import os
from pathlib import Path

ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://ml-service:8001")
OPTIMIZER_SERVICE_URL = os.environ.get("OPTIMIZER_SERVICE_URL", "http://optimizer:8002")
ML_TIMEOUT_S = float(os.environ.get("ML_TIMEOUT_S", "5.0"))
OPTIMIZER_TIMEOUT_S = float(os.environ.get("OPTIMIZER_TIMEOUT_S", "30.0"))

# Shared data directory (mounted volume for both gateway and ml-service)
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
