import json
from pathlib import Path


def log_watch(path, report):
    if report is None:
        return
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report) + "\n")
    print(f"[parameter-watch] {report}")
