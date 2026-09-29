from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def ensure_run_dirs(out_dir: str | Path) -> dict[str, Path]:
    root = Path(out_dir).expanduser().resolve()
    dirs = {
        "root": root,
        "manifests": root / "manifests",
        "cleaned": root / "cleaned" / "af2rank_input",
        "af2rank": root / "af2rank",
        "af2rank_pae": root / "af2rank" / "pae",
        "logs": root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_csv(path: str | Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def completed_ids_from_jsonl(path: str | Path, stage: str | None = None) -> set[str]:
    completed = set()
    for record in read_jsonl(path):
        if record.get("status") == "ok" and (stage is None or record.get("stage") == stage):
            completed.add(record["model_id"])
    return completed


def read_csv_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_failure(path: str | Path, stage: str, model_id: str, source_path: str, exc: Exception, suggestion: str) -> None:
    append_jsonl(
        path,
        {
            "stage": stage,
            "model_id": model_id,
            "source_path": source_path,
            "exception_class": exc.__class__.__name__,
            "message": str(exc),
            "suggested_action": suggestion,
        },
    )
