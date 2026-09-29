from __future__ import annotations

import os
from pathlib import Path

from . import dockq_pae
from .manifests import ensure_run_dirs, read_csv_rows, write_failure, write_jsonl
from .target import TargetSpec


def bundled_dockq_src() -> Path:
    package_root = Path(__file__).resolve().parent.parent
    checkout_src = package_root / "DockQ" / "src"
    if (checkout_src / "DockQ" / "__init__.py").is_file():
        return checkout_src
    if (package_root / "DockQ" / "__init__.py").is_file():
        return package_root
    raise FileNotFoundError("Modified DockQ source is missing from this installation")


def resolve_dockq_src(dockq_src: str | Path | None) -> Path:
    return Path(dockq_src) if dockq_src else bundled_dockq_src()


def _prepend_pythonpath(path: str | Path | None) -> str | None:
    if not path:
        return None
    old_value = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(path) + (os.pathsep + old_value if old_value else "")
    return old_value


def _restore_pythonpath(old_value: str | None) -> None:
    if old_value is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = old_value


def run_dockq_io_stage(
    target_spec: TargetSpec,
    out_dir: str | Path,
    *,
    dockq_src: str | Path | None = None,
    jobs: int = 1,
) -> list[dict]:
    dirs = ensure_run_dirs(out_dir)
    manifest_csv = dirs["manifests"] / "dockq_pae_manifest.csv"
    metrics_dir = dirs["root"] / "additional_metrics"
    failures = dirs["logs"] / "failures.jsonl"
    stage_manifest = dirs["manifests"] / "dockq_io.jsonl"

    if not manifest_csv.is_file():
        raise FileNotFoundError(
            f"Additional-metrics manifest not found: {manifest_csv}. Run the af2rank stage first."
        )

    try:
        specs = dockq_pae.read_manifest(manifest_csv)
        active_dockq_src = resolve_dockq_src(dockq_src)
        old_pythonpath = _prepend_pythonpath(active_dockq_src)
        try:
            dockq_pae.process_specs(specs, metrics_dir, jobs=jobs)
        finally:
            _restore_pythonpath(old_pythonpath)
    except Exception as exc:
        write_failure(
            failures,
            "dockq_io",
            target_spec.target,
            str(manifest_csv),
            exc,
            "Check DockQ installation, AF2Rank scored PDBs, PAE JSONs, and chain mapping.",
        )
        raise

    records_by_model: dict[str, dict] = {}
    for row in read_csv_rows(metrics_dir / "dockq_metrics.csv"):
        model_id = Path(row.get("model_id", "")).stem
        if not model_id:
            continue
        records_by_model.setdefault(
            model_id,
            {
                "stage": "dockq_io",
                "status": "ok",
                "model_id": model_id,
                "dockq_json": str(metrics_dir / "dockq_json" / f"{model_id}.json"),
                "dockq_io": row.get("GlobalDockQ") or row.get("best_dockq") or "",
                "best_mapping": row.get("best_mapping_str", ""),
                "n_interfaces": 0,
                "dockq_metrics_csv": str(metrics_dir / "dockq_metrics.csv"),
                "pae_raw_csv": str(metrics_dir / "pae_raw.csv.gz"),
            },
        )
        records_by_model[model_id]["n_interfaces"] += 1

    records = [records_by_model[key] for key in sorted(records_by_model)]
    expected_ids = {Path(spec.model_id).stem for spec in specs}
    if set(records_by_model) != expected_ids:
        missing = sorted(expected_ids - set(records_by_model))
        extra = sorted(set(records_by_model) - expected_ids)
        exc = ValueError(
            f"DockQ results do not match inputs: {len(missing)} missing, "
            f"{len(extra)} unexpected model(s); missing={missing[:5]}, unexpected={extra[:5]}"
        )
        write_failure(
            failures,
            "dockq_io",
            target_spec.target,
            str(manifest_csv),
            exc,
            "Check DockQ outputs and the input manifest before aggregating.",
        )
        raise exc
    write_jsonl(stage_manifest, records)
    return records
