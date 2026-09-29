from __future__ import annotations

import shutil
from pathlib import Path

from .ingest import RAW_MODEL_FIELDS, RawModel, discover_models, read_raw_models_manifest, write_raw_models_manifest
from .manifests import write_csv
from .target import load_target_spec


def _validate_num_batches(num_batches: int, model_count: int) -> None:
    if num_batches < 1:
        raise ValueError("--num-batches must be at least 1")
    if model_count < 1:
        raise ValueError("No supported model files were discovered")
    if num_batches > model_count:
        raise ValueError(f"--num-batches ({num_batches}) cannot exceed model count ({model_count})")


def _partition(raw_models: list[RawModel], num_batches: int) -> list[list[RawModel]]:
    _validate_num_batches(num_batches, len(raw_models))
    base_size, remainder = divmod(len(raw_models), num_batches)
    batches = []
    start = 0
    for batch_idx in range(num_batches):
        size = base_size + (1 if batch_idx < remainder else 0)
        batches.append(raw_models[start : start + size])
        start += size
    return batches


def write_raw_batch_manifest(batch_root: Path, batch_idx: int, raw_models: list[RawModel]) -> Path:
    path = batch_root / f"batch_{batch_idx:03d}.csv"
    write_csv(path, [model.to_row() for model in raw_models], RAW_MODEL_FIELDS)
    return path


def prepare_raw_batches(
    target: str,
    fasta: str | Path,
    models_path: str | Path,
    out_dir: str | Path,
    batch_root: str | Path,
    *,
    num_batches: int,
    overwrite: bool = False,
) -> int:
    out_dir = Path(out_dir)
    batch_root = Path(batch_root)
    target_spec = load_target_spec(target, fasta)
    target_spec.write_json(out_dir / "target_spec.json")

    raw_models = discover_models(models_path)
    write_raw_models_manifest(raw_models, out_dir / "manifests" / "raw_models.csv")

    if batch_root.exists():
        if not overwrite:
            raise FileExistsError(f"Batch root already exists: {batch_root}")
        shutil.rmtree(batch_root)
    batch_root.mkdir(parents=True)

    for batch_idx, batch_models in enumerate(_partition(raw_models, num_batches)):
        write_raw_batch_manifest(batch_root, batch_idx, batch_models)
    return num_batches


def read_raw_batch_manifest(path: str | Path) -> list[RawModel]:
    return read_raw_models_manifest(path)


def materialize_cleaned_batch(
    target: str,
    out_dir: str | Path,
    batch_manifest: str | Path,
    dest_dir: str | Path,
) -> int:
    cleaned_dir = Path(out_dir).expanduser().resolve() / "cleaned" / "af2rank_input" / target
    dest_dir = Path(dest_dir)
    raw_models = read_raw_batch_manifest(batch_manifest)
    if not raw_models:
        raise ValueError(f"Batch manifest contains no models: {batch_manifest}")
    missing = [
        raw_model.model_id
        for raw_model in raw_models
        if not (cleaned_dir / f"{raw_model.model_id}.pdb").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing cleaned PDBs for {len(missing)} batch model(s): {', '.join(missing[:5])}"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for raw_model in raw_models:
        src = cleaned_dir / f"{raw_model.model_id}.pdb"
        shutil.copy2(src, dest_dir / src.name)
        count += 1
    return count
