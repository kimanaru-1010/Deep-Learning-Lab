import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def create_run(root="outputs/runs", name=None, config=None):
    if name is None:
        name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Run name must be a simple filename, without path separators")
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=False)
    (path / "checkpoints").mkdir()
    (path / "config.json").write_text(json.dumps(config or {}, indent=2), encoding="utf-8")
    return path
