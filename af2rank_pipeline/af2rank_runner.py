from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .manifests import ensure_run_dirs, read_jsonl, write_csv, write_failure
from .target import TargetSpec

AF2RANK_FIELDS = [
    "model_id",
    "status",
    "input_pdb",
    "output_pdb",
    "pae_json",
    "raw_npz",
    "af2rank_csv",
    "target",
    "model_name",
    "model_mode",
    "version",
    "model_num",
    "recycle",
    "iterations",
    "chain",
    "rm_seq",
    "rm_sc",
    "rm_ic",
    "plddt",
    "ptm",
    "i_ptm",
    "tm_i",
    "tm_o",
    "tm_io",
    "composite",
    "pae",
]

DOCKQ_PAE_MANIFEST_FIELDS = [
    "name",
    "batch",
    "model_id",
    "native",
    "model",
    "pae",
    "af2rank_csv",
    "mapping",
    "pae_chain_order",
]


@dataclass
class AF2RankConfig:
    """Configuration for the ActifPTM AF2Rank implementation.

    This stage runs the bundled ActifPTM/AF2Rank implementation for cleaned
    models and records outputs for interface analysis.
    """

    params: str = ""
    colabdesign: str | None = None
    tm_exec: str | None = None
    model_mode: str = "alphafold"
    version: str = "ptm"
    model_num: int = 2
    recycle: int = 1
    iterations: int = 1
    mask_sequence: bool = True
    mask_sidechains: bool = True
    mask_interchain: bool = False
    keep_raw: bool = False
    pae_json_gzip: bool = True
    pipeline_script: str | None = None
    python_executable: str = sys.executable
    extra_args: list[str] = field(default_factory=list)
    decoy_dir: str | None = None
    run_label: str | None = None
    partial_manifest: bool = False


def default_actifptm_script() -> Path:
    return Path(__file__).with_name("actifptm_runner.py")


def model_subdir(cfg: AF2RankConfig) -> str:
    return f"{cfg.model_mode}_{cfg.version}_model{cfg.model_num}_rec{cfg.recycle}"


def run_label(target_spec: TargetSpec, cfg: AF2RankConfig) -> str:
    return cfg.run_label or target_spec.target


def run_output_dir(dirs: dict[str, Path], cfg: AF2RankConfig) -> Path:
    base = dirs["af2rank"]
    if cfg.run_label:
        base = base / cfg.run_label
    return base / model_subdir(cfg)


def manifest_paths(dirs: dict[str, Path], cfg: AF2RankConfig) -> tuple[Path, Path]:
    if cfg.partial_manifest:
        if not cfg.run_label:
            raise ValueError("partial manifests require --run-label")
        suffix = cfg.run_label
        return (
            dirs["manifests"] / f"af2rank_{suffix}.csv",
            dirs["manifests"] / f"dockq_pae_manifest_{suffix}.csv",
        )
    return dirs["manifests"] / "af2rank.csv", dirs["manifests"] / "dockq_pae_manifest.csv"


def collect_fieldnames(preferred: list[str], rows: list[dict]) -> list[str]:
    fieldnames = list(preferred)
    seen = set(fieldnames)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def _run_actifptm_script(
    target_spec: TargetSpec,
    decoy_dir: Path,
    output_dir: Path,
    pae_json_dir: Path,
    cfg: AF2RankConfig,
) -> None:
    if not cfg.pipeline_script and not cfg.params:
        raise ValueError("AF2Rank inference requires an AlphaFold parameter path")
    script = Path(cfg.pipeline_script) if cfg.pipeline_script else default_actifptm_script()
    cmd = [
        cfg.python_executable,
        str(script),
        "--decoy_dir",
        str(decoy_dir),
        "--chain",
        ",".join(target_spec.chain_ids),
        "--model_mode",
        cfg.model_mode,
        "--model_num",
        str(cfg.model_num),
        "--recycle",
        str(cfg.recycle),
        "--iterations",
        str(cfg.iterations),
        "--params",
        cfg.params,
        "--output_dir",
        str(output_dir),
        "--pae_json_dir",
        str(pae_json_dir),
        "--target",
        run_label(target_spec, cfg),
    ]
    if cfg.colabdesign:
        cmd.extend(["--colabdesign", cfg.colabdesign])
    if cfg.tm_exec:
        cmd.extend(["--tm", cfg.tm_exec])
    if cfg.mask_sequence:
        cmd.append("--mask_sequence")
    if cfg.mask_sidechains:
        cmd.append("--mask_sidechains")
    if cfg.mask_interchain:
        cmd.append("--mask_interchain")
    if cfg.keep_raw:
        cmd.append("--keep_raw")
    if not cfg.pae_json_gzip:
        cmd.append("--no_pae_json_gzip")
    cmd.extend(cfg.extra_args)

    subprocess.run(cmd, check=True)


def _write_pae_json(pae_json: Path, model_id: str, pae: np.ndarray) -> None:
    pae_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_id": model_id, "pae": np.asarray(pae, dtype=float).tolist()}
    if pae_json.suffix == ".gz":
        with gzip.open(pae_json, "wt") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.write("\n")
    else:
        pae_json.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def _npz_to_pae_json(raw_npz: Path, pae_json: Path, model_id: str) -> None:
    with np.load(raw_npz) as data:
        if "pae" not in data:
            raise ValueError(f"{raw_npz} does not contain a 'pae' array")
        pae = np.asarray(data["pae"], dtype=float)
    _write_pae_json(pae_json, model_id, pae)


def _read_af2rank_csv(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"ActifPTM score CSV was not written: {csv_path}")
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _resume_config(target_spec: TargetSpec, cfg: AF2RankConfig, decoy_dir: Path) -> dict:
    def resolved(value: str | None) -> str:
        return str(Path(value).expanduser().resolve()) if value else ""

    return {
        "target": target_spec.target,
        "chain_ids": target_spec.chain_ids,
        "sequence": target_spec.flat_sequence,
        "decoy_dir": str(decoy_dir.expanduser().resolve()),
        "params": resolved(cfg.params),
        "colabdesign": resolved(cfg.colabdesign),
        "tm_exec": resolved(cfg.tm_exec),
        "pipeline_script": resolved(cfg.pipeline_script),
        "python_executable": cfg.python_executable,
        "model_mode": cfg.model_mode,
        "version": cfg.version,
        "model_num": cfg.model_num,
        "recycle": cfg.recycle,
        "iterations": cfg.iterations,
        "mask_sequence": cfg.mask_sequence,
        "mask_sidechains": cfg.mask_sidechains,
        "mask_interchain": cfg.mask_interchain,
        "keep_raw": cfg.keep_raw,
        "pae_json_gzip": cfg.pae_json_gzip,
        "extra_args": list(cfg.extra_args),
        "run_label": cfg.run_label,
        "partial_manifest": cfg.partial_manifest,
    }


def _find_pae_json(pae_dir: Path, model_id: str, *, prefer_gzip: bool = True) -> Path:
    preferred = pae_dir / f"{model_id}.json.gz" if prefer_gzip else pae_dir / f"{model_id}.json"
    fallback = pae_dir / f"{model_id}.json" if prefer_gzip else pae_dir / f"{model_id}.json.gz"
    if preferred.is_file():
        return preferred
    if fallback.is_file():
        return fallback
    return preferred


def _write_stage_manifests(
    target_spec: TargetSpec,
    out_dir: Path,
    cfg: AF2RankConfig,
) -> list[dict]:
    dirs = ensure_run_dirs(out_dir)
    run_dir = run_output_dir(dirs, cfg)
    label = run_label(target_spec, cfg)
    score_csv = run_dir / f"{label}_af2rank_actifptm.csv"
    scored_dir = run_dir / "scored_pdbs"
    raw_dir = run_dir / "raw_npz"
    pae_dir = dirs["af2rank_pae"]
    cleaning_by_id = {
        row["model_id"]: row
        for row in read_jsonl(dirs["manifests"] / "cleaning.jsonl")
        if row.get("status") == "ok"
    }

    records: list[dict] = []
    dockq_manifest_rows: list[dict] = []
    mapping = f"{target_spec.chain_ids}:{target_spec.chain_ids}"
    chain_order = ",".join(target_spec.chain_ids)
    decoy_dir = Path(cfg.decoy_dir) if cfg.decoy_dir else dirs["cleaned"] / target_spec.target
    expected_ids = {path.stem for path in decoy_dir.glob("*.pdb")}
    score_rows = _read_af2rank_csv(score_csv)
    scored_ids = [Path(row.get("id") or row.get("model_id") or "").stem for row in score_rows]
    if not expected_ids or len(scored_ids) != len(expected_ids) or set(scored_ids) != expected_ids:
        missing = sorted(expected_ids - set(scored_ids))
        extra = sorted(set(scored_ids) - expected_ids)
        raise ValueError(
            f"AF2Rank results do not match cleaned inputs: {len(expected_ids)} input(s), "
            f"{len(scored_ids)} score row(s); missing={missing[:5]}, extra={extra[:5]}. "
            "Use a fresh output directory after changing inputs."
        )

    settings = {
        "model_mode": cfg.model_mode,
        "version": cfg.version,
        "model_num": cfg.model_num,
        "recycle": cfg.recycle,
        "iterations": cfg.iterations,
        "rm_seq": cfg.mask_sequence,
        "rm_sc": cfg.mask_sidechains,
        "rm_ic": cfg.mask_interchain,
    }
    for score_row in score_rows:
        for key, expected in settings.items():
            actual = score_row.get(key)
            if actual not in (None, "") and str(actual).lower() != str(expected).lower():
                raise ValueError(
                    f"AF2Rank score CSV has {key}={actual}, but this run requests {expected}. "
                    "Use a fresh output directory after changing settings."
                )
        input_name = score_row.get("id") or score_row.get("model_id") or ""
        if not input_name:
            continue
        model_id = Path(input_name).stem
        clean_record = cleaning_by_id.get(model_id)
        if clean_record is None:
            raise ValueError(f"{score_csv} contains {input_name}, but cleaning manifest has no model_id={model_id}")
        input_pdb = Path(clean_record["cleaned_path"])
        output_pdb = scored_dir / f"{model_id}_scored.pdb"
        raw_npz = raw_dir / f"{model_id}_raw_outputs.npz"
        pae_json = _find_pae_json(pae_dir, model_id, prefer_gzip=cfg.pae_json_gzip)
        if not output_pdb.is_file():
            raise FileNotFoundError(f"ActifPTM scored PDB not found: {output_pdb}")
        if not pae_json.is_file() and raw_npz.is_file():
            _npz_to_pae_json(raw_npz, pae_json, model_id)
        if not pae_json.is_file():
            raise FileNotFoundError(
                f"ActifPTM PAE JSON not found: {pae_json}. Check PAE export from the AF2Rank batch."
            )

        record = {
            "model_id": model_id,
            "status": "ok",
            "input_pdb": str(input_pdb),
            "output_pdb": str(output_pdb),
            "pae_json": str(pae_json),
            "raw_npz": str(raw_npz) if raw_npz.is_file() else "",
            "af2rank_csv": str(score_csv),
            **score_row,
        }
        records.append(record)
        dockq_manifest_rows.append(
            {
                "name": model_id,
                "batch": label,
                "model_id": input_name,
                "native": str(input_pdb),
                "model": str(output_pdb),
                "pae": str(pae_json),
                "af2rank_csv": str(score_csv),
                "mapping": mapping,
                "pae_chain_order": chain_order,
            }
        )

    af2rank_manifest, dockq_manifest = manifest_paths(dirs, cfg)
    write_csv(af2rank_manifest, records, collect_fieldnames(AF2RANK_FIELDS, records))
    write_csv(
        dockq_manifest,
        dockq_manifest_rows,
        collect_fieldnames(DOCKQ_PAE_MANIFEST_FIELDS, dockq_manifest_rows),
    )
    return records


def run_af2rank_stage(
    target_spec: TargetSpec,
    out_dir: str | Path,
    cfg: AF2RankConfig,
    *,
    resume: bool = False,
) -> list[dict]:
    dirs = ensure_run_dirs(out_dir)
    decoy_dir = Path(cfg.decoy_dir) if cfg.decoy_dir else dirs["cleaned"] / target_spec.target
    run_dir = run_output_dir(dirs, cfg)
    failures = dirs["logs"] / "failures.jsonl"

    try:
        if cfg.iterations < 1:
            raise ValueError("iterations must be at least 1")
        expected_version = "v3" if cfg.model_mode == "alphafold-multimer" else "ptm"
        if cfg.version != expected_version:
            raise ValueError(f"{cfg.model_mode} requires version {expected_version}")
        af2rank_manifest, _ = manifest_paths(dirs, cfg)
        config_path = run_dir / "pipeline_run_config.json"
        current_config = _resume_config(target_spec, cfg, decoy_dir)
        if resume and af2rank_manifest.is_file():
            if not config_path.is_file():
                raise ValueError(
                    f"Cannot verify AF2Rank resume settings: {config_path} is missing. "
                    "Use a fresh output directory or rerun without --resume."
                )
            saved_config = json.loads(config_path.read_text())
            if saved_config != current_config:
                changed = sorted(
                    key for key in set(saved_config) | set(current_config)
                    if saved_config.get(key) != current_config.get(key)
                )
                raise ValueError(
                    f"Cannot resume AF2Rank: scoring settings changed ({', '.join(changed)}). "
                    "Use a fresh output directory or rerun without --resume."
                )
            return _write_stage_manifests(target_spec, Path(out_dir), cfg)
        _run_actifptm_script(target_spec, decoy_dir, run_dir, dirs["af2rank_pae"], cfg)
        records = _write_stage_manifests(target_spec, Path(out_dir), cfg)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(current_config, indent=2) + "\n")
        return records
    except Exception as exc:
        write_failure(
            failures,
            "af2rank",
            target_spec.target,
            str(decoy_dir),
            exc,
            "Check the ColabDesign environment, AlphaFold params path, GPU/JAX runtime, and cleaned PDB directory.",
        )
        raise


def merge_af2rank_batch_manifests(out_dir: str | Path) -> tuple[list[dict], list[dict]]:
    dirs = ensure_run_dirs(out_dir)
    partial_af2rank = sorted(
        path for path in dirs["manifests"].glob("af2rank_*.csv") if path.name != "af2rank.csv"
    )
    partial_dockq = sorted(
        path
        for path in dirs["manifests"].glob("dockq_pae_manifest_*.csv")
        if path.name != "dockq_pae_manifest.csv"
    )
    if not partial_af2rank:
        raise FileNotFoundError(f"No partial AF2Rank manifests found under {dirs['manifests']}")
    if not partial_dockq:
        raise FileNotFoundError(f"No partial DockQ/PAE manifests found under {dirs['manifests']}")

    af2rank_rows: list[dict] = []
    for path in partial_af2rank:
        af2rank_rows.extend(_read_af2rank_csv(path))
    dockq_rows: list[dict] = []
    for path in partial_dockq:
        dockq_rows.extend(_read_af2rank_csv(path))

    write_csv(
        dirs["manifests"] / "af2rank.csv",
        af2rank_rows,
        collect_fieldnames(AF2RANK_FIELDS, af2rank_rows),
    )
    write_csv(
        dirs["manifests"] / "dockq_pae_manifest.csv",
        dockq_rows,
        collect_fieldnames(DOCKQ_PAE_MANIFEST_FIELDS, dockq_rows),
    )
    return af2rank_rows, dockq_rows
