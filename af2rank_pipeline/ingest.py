from __future__ import annotations

import gzip
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from .manifests import write_csv

SUPPORTED_MODEL_SUFFIXES = {".pdb", ".cif", ".mmcif"}
RAW_MODEL_FIELDS = ["model_id", "source_path", "source_format", "model_number", "capri_md5"]


@dataclass(frozen=True)
class RawModel:
    model_id: str
    source_path: str
    source_format: str
    model_number: int | None = None
    capri_md5: str | None = None

    def to_row(self) -> dict:
        row = asdict(self)
        row["model_number"] = "" if self.model_number is None else self.model_number
        row["capri_md5"] = self.capri_md5 or ""
        return row


def sanitize_model_id(value: str) -> str:
    value = re.sub(r"\.(pdb|cif|mmcif)(\.gz)?$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or hashlib.sha1(value.encode()).hexdigest()[:12]


def model_suffix(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".gz":
        inner_suffix = Path(path.stem).suffix.lower()
        return inner_suffix if inner_suffix == ".pdb" else ""
    return suffix


def _open_model_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open(errors="replace")


def _model_records_in_pdb(path: Path) -> list[int]:
    records = []
    with _open_model_text(path) as handle:
        for line in handle:
            if line.startswith("MODEL "):
                try:
                    records.append(int(line[6:14].strip()))
                except ValueError:
                    records.append(len(records) + 1)
    return records


def _capri_md5_by_model(path: Path) -> dict[int, str]:
    md5_by_model = {}
    pattern = re.compile(r"REMARK\s+9\s+MODEL\s+(\d+)\s+MD5\s+([0-9a-fA-F]{32})")
    with _open_model_text(path) as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                md5_by_model[int(match.group(1))] = match.group(2).lower()
    return md5_by_model


def discover_models(models_path: str | Path) -> list[RawModel]:
    root = Path(models_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Models path does not exist: {root}")

    files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    raw_models: list[RawModel] = []
    for path in files:
        suffix = model_suffix(path)
        if suffix not in SUPPORTED_MODEL_SUFFIXES:
            continue
        source_format = suffix.lstrip(".")
        if suffix == ".pdb":
            model_numbers = _model_records_in_pdb(path)
            if len(model_numbers) > 1:
                md5_by_model = _capri_md5_by_model(path)
                for model_number in model_numbers:
                    raw_models.append(
                        RawModel(
                            model_id=f"{sanitize_model_id(path.name)}_model_{model_number:04d}",
                            source_path=str(path),
                            source_format=source_format,
                            model_number=model_number,
                            capri_md5=md5_by_model.get(model_number),
                        )
                    )
                continue
        raw_models.append(
            RawModel(
                model_id=sanitize_model_id(path.name),
                source_path=str(path),
                source_format=source_format,
                model_number=None,
                capri_md5=None,
            )
        )

    return raw_models


def write_raw_models_manifest(raw_models: list[RawModel], path: str | Path) -> None:
    write_csv(path, [model.to_row() for model in raw_models], RAW_MODEL_FIELDS)


def read_raw_models_manifest(path: str | Path) -> list[RawModel]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw model manifest does not exist: {path}")
    import csv

    with path.open(newline="") as handle:
        return [raw_model_from_row(row) for row in csv.DictReader(handle)]


def raw_model_from_row(row: dict) -> RawModel:
    model_number = row.get("model_number")
    return RawModel(
        model_id=row["model_id"],
        source_path=row["source_path"],
        source_format=row["source_format"],
        model_number=int(model_number) if model_number else None,
        capri_md5=row.get("capri_md5") or None,
    )
