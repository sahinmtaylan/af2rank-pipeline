from __future__ import annotations

import gzip
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from .exceptions import CleaningError, DependencyError
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
    open_model_line: int | None = None
    first_unwrapped_atom: int | None = None

    def invalid(reason: str) -> None:
        raise CleaningError(
            f"{path.name}: {reason}. Each ensemble member must start with a numbered "
            "MODEL record and end with ENDMDL. Add the missing markers or upload "
            "the structures as separate PDB files."
        )

    with _open_model_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            record = line[:6].strip()
            if record == "MODEL":
                if open_model_line is not None:
                    invalid(f"MODEL at line {line_number} appears before ENDMDL for MODEL at line {open_model_line}")
                if first_unwrapped_atom is not None:
                    invalid(f"Atoms at line {first_unwrapped_atom} appear before the first MODEL header")
                try:
                    model_number = int(line[6:14].strip())
                except ValueError:
                    invalid(f"MODEL at line {line_number} has no valid integer model number")
                records.append(model_number)
                open_model_line = line_number
            elif record == "ENDMDL":
                if open_model_line is None:
                    invalid(f"ENDMDL at line {line_number} has no preceding MODEL header")
                open_model_line = None
            elif record in {"ATOM", "HETATM"} and open_model_line is None:
                if records:
                    invalid(f"Atoms at line {line_number} appear outside a MODEL/ENDMDL block")
                if first_unwrapped_atom is None:
                    first_unwrapped_atom = line_number
            elif record == "END" and open_model_line is not None:
                invalid(f"MODEL at line {open_model_line} has no ENDMDL before END at line {line_number}")
    if open_model_line is not None:
        invalid(f"MODEL at line {open_model_line} has no ENDMDL before the end of the file")
    return records


def _model_records_in_mmcif(path: Path) -> list[int]:
    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    except ImportError as exc:
        raise DependencyError("mmCIF parsing requires biopython. Install the package dependencies.") from exc

    data = MMCIF2Dict(str(path))
    try:
        return list(dict.fromkeys(int(value) for value in data.get("_atom_site.pdbx_PDB_model_num", [])))
    except ValueError as exc:
        raise CleaningError(f"{path.name}: _atom_site.pdbx_PDB_model_num contains an invalid model number") from exc


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
        model_numbers = (
            _model_records_in_pdb(path) if suffix == ".pdb" else _model_records_in_mmcif(path)
        )
        if len(model_numbers) > 1:
            md5_by_model = _capri_md5_by_model(path) if suffix == ".pdb" else {}
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
