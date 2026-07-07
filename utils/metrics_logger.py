"""Metric persistence utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class MetricsLogger:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "metrics.csv"

    def write_jsonl(self, name: str, row: dict[str, Any]) -> None:
        with (self.log_dir / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def update_csv(self, row: dict[str, Any]) -> None:
        exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def write_summary(self, row: dict[str, Any]) -> None:
        with (self.log_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)
