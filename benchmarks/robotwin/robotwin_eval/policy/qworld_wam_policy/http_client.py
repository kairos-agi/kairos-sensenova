from __future__ import annotations

import sys
from pathlib import Path

_KAIROS_WAM_ROOT = Path(__file__).resolve().parents[5]
if str(_KAIROS_WAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_KAIROS_WAM_ROOT))

from benchmarks.common.clients.wam_http_client import WAMServiceClient

__all__ = ["WAMServiceClient"]
