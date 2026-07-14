"""Small JSONL metrics sink."""

import json
from pathlib import Path
from typing import Any


class MetricsWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, values: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, ensure_ascii=False, sort_keys=True) + "\n")
