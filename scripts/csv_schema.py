#!/usr/bin/env python3
"""Canonical 22-column CSV writer enforcing Ground Rule #4 across all TabICLv2 TorchTPU phases."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

CSV_COLUMNS: list[str] = [
    "phase",
    "path",
    "model",
    "dtype",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "n_estimators",
    "kv_cache",
    "devices",
    "batch_datasets",
    "compile_s",
    "p50_ms",
    "p99_ms",
    "datasets_per_s",
    "test_rows_per_s",
    "peak_hbm_gib",
    "metric_name",
    "metric_value",
    "ref_metric_value",
    "notes",
]


def write_rows(csv_path: str | Path, rows: Iterable[Mapping[str, Any]], append: bool = False) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not append) or (not path.exists()) or path.stat().st_size == 0
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="raise")
        if write_header:
            writer.writeheader()
        for r in rows:
            normalized = {col: r.get(col, "") for col in CSV_COLUMNS}
            writer.writerow(normalized)
