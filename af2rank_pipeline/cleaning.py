from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

from .alignment import PairwiseChainAlignment, align_chain_to_target, assign_chains
from .exceptions import CleaningError
from .batching import read_raw_batch_manifest
from .ingest import RawModel, discover_models, model_suffix, read_raw_models_manifest, write_raw_models_manifest
from .manifests import append_jsonl, completed_ids_from_jsonl, ensure_run_dirs, read_jsonl, write_failure, write_jsonl
from .structure import (
    CleanResidue,
    ONE_TO_THREE,
    StructureRecord,
    iter_pdb_model_lines,
    load_structure,
    parse_pdb_lines,
    write_clean_pdb,
)
from .target import TargetSpec, load_target_spec_json


def _alignment_to_record(alignment: PairwiseChainAlignment) -> dict:
    return {
        "raw_chain_id": alignment.raw_chain_id,
        "target_chain_id": alignment.target_chain_id,
        "method": alignment.method,
        "identity": alignment.identity,
        "raw_coverage": alignment.raw_coverage,
        "target_coverage": alignment.target_coverage,
        "score": alignment.score,
        "matches": alignment.matches,
        "mismatches": alignment.mismatches,
        "raw_aligned": alignment.raw_aligned,
        "target_aligned": alignment.target_aligned,
    }


def _fmt_chain_summary(values: dict[str, object]) -> str:
    return ";".join(f"{chain}:{value}" for chain, value in sorted(values.items()))


def _chain_alignment_summary(alignments: list[PairwiseChainAlignment]) -> dict[str, str]:
    methods: dict[str, str] = {}
    raw_coverages: dict[str, str] = {}
    target_coverages: dict[str, str] = {}
    target_starts: dict[str, str] = {}
    target_ends: dict[str, str] = {}
    n_terminal_missing: dict[str, str] = {}
    c_terminal_missing: dict[str, str] = {}

    for alignment in alignments:
        target = alignment.target_chain_id
        matched_map = alignment.matched_residue_map
        target_indices = sorted(matched_map.values())
        methods[target] = alignment.method
        raw_coverages[target] = f"{alignment.raw_coverage:.3f}"
        target_coverages[target] = f"{alignment.target_coverage:.3f}"
        if target_indices:
            start = target_indices[0] + 1
            end = target_indices[-1] + 1
            target_starts[target] = str(start)
            target_ends[target] = str(end)
            n_terminal_missing[target] = str(start - 1)
            c_terminal_missing[target] = str(len(alignment.target_sequence) - end)
        else:
            target_starts[target] = ""
            target_ends[target] = ""
            n_terminal_missing[target] = str(len(alignment.target_sequence))
            c_terminal_missing[target] = str(len(alignment.target_sequence))

    return {
        "chain_alignment_methods": _fmt_chain_summary(methods),
        "chain_raw_coverages": _fmt_chain_summary(raw_coverages),
        "chain_target_coverages": _fmt_chain_summary(target_coverages),
        "chain_target_starts": _fmt_chain_summary(target_starts),
        "chain_target_ends": _fmt_chain_summary(target_ends),
        "chain_n_terminal_missing": _fmt_chain_summary(n_terminal_missing),
        "chain_c_terminal_missing": _fmt_chain_summary(c_terminal_missing),
    }


def _clean_structure(
    raw_model: RawModel,
    structure: StructureRecord,
    target_spec: TargetSpec,
    output_pdb: str | Path,
    *,
    min_identity: float = 0.90,
    min_raw_coverage: float = 0.85,
    min_target_coverage: float = 0.50,
) -> dict:
    raw_chains = [chain for chain in structure.chains if chain.residues]
    if not raw_chains:
        raise CleaningError("No protein chains found in model")

    alignments = assign_chains(
        raw_chains,
        target_spec.chains,
        min_identity=min_identity,
        min_raw_coverage=min_raw_coverage,
        min_target_coverage=min_target_coverage,
    )
    assigned_raw_chain_ids = {alignment.raw_chain_id for alignment in alignments}
    matching_unassigned = []
    for raw_chain in raw_chains:
        if raw_chain.chain_id in assigned_raw_chain_ids:
            continue
        for target_chain in target_spec.chains:
            alignment = align_chain_to_target(raw_chain, target_chain)
            if (
                alignment.identity >= min_identity
                and alignment.raw_coverage >= min_raw_coverage
                and alignment.target_coverage >= min_target_coverage
            ):
                matching_unassigned.append(raw_chain.chain_id)
                break
    if matching_unassigned:
        raise CleaningError(
            "Model contains unassigned protein chains that match the target FASTA: "
            + ",".join(sorted(matching_unassigned))
            + ". The FASTA may omit homomer copies from the modeled assembly."
        )

    alignment_by_target = {alignment.target_chain_id: alignment for alignment in alignments}
    raw_by_id = {chain.chain_id: chain for chain in raw_chains}

    clean_residues: list[CleanResidue] = []
    skipped_mismatches = 0
    for target_chain in target_spec.chains:
        alignment = alignment_by_target[target_chain.canonical_id]
        raw_chain = raw_by_id[alignment.raw_chain_id]
        matched_map = alignment.matched_residue_map
        skipped_mismatches += len(alignment.residue_map) - len(matched_map)

        for raw_idx, target_idx in sorted(matched_map.items(), key=lambda item: item[1]):
            residue = raw_chain.residues[raw_idx]
            flat_index = target_chain.offset + target_idx + 1
            clean_residues.append(
                CleanResidue(
                    target_chain_id=target_chain.canonical_id,
                    flat_index=flat_index,
                    local_index=target_idx + 1,
                    resname=ONE_TO_THREE[target_chain.sequence[target_idx]],
                    source_chain_id=residue.chain_id,
                    source_resseq=residue.resseq,
                    source_icode=residue.icode,
                    atoms=residue.atoms,
                )
            )

    if not clean_residues:
        raise CleaningError("No residues survived target mapping")

    clean_residues.sort(key=lambda residue: residue.flat_index)
    write_clean_pdb(clean_residues, output_pdb)

    covered_indices = {residue.flat_index for residue in clean_residues}
    expected_indices = set(range(1, target_spec.length + 1))
    missing = sorted(expected_indices - covered_indices)

    return {
        "stage": "clean",
        "status": "ok",
        "model_id": raw_model.model_id,
        "source_path": raw_model.source_path,
        "source_format": raw_model.source_format,
        "model_number": raw_model.model_number,
        "capri_md5": raw_model.capri_md5,
        "cleaned_path": str(output_pdb),
        "target_length": target_spec.length,
        "cleaned_residue_count": len(clean_residues),
        "missing_residue_count": len(missing),
        "missing_residue_indices": missing[:200],
        "skipped_mismatch_count": skipped_mismatches,
        "chain_mapping": [_alignment_to_record(alignment) for alignment in alignments],
        **_chain_alignment_summary(alignments),
    }


def clean_model(
    raw_model: RawModel,
    target_spec: TargetSpec,
    output_pdb: str | Path,
    *,
    min_identity: float = 0.90,
    min_raw_coverage: float = 0.85,
    min_target_coverage: float = 0.50,
) -> dict:
    structure = load_structure(raw_model.source_path, model_number=raw_model.model_number)
    return _clean_structure(
        raw_model,
        structure,
        target_spec,
        output_pdb,
        min_identity=min_identity,
        min_raw_coverage=min_raw_coverage,
        min_target_coverage=min_target_coverage,
    )


def _complete_clean_records(
    raw_models: list[RawModel], manifest: Path, failed_ids: list[str], failures: Path,
    *, strict_clean: bool = False,
) -> list[dict]:
    if not raw_models:
        raise CleaningError("No supported models were found in the input")
    successful = {
        row["model_id"]: row
        for row in read_jsonl(manifest)
        if row.get("stage") == "clean"
        and row.get("status") == "ok"
        and row.get("cleaned_path")
        and Path(row["cleaned_path"]).is_file()
        and row["model_id"] not in failed_ids
    }
    missing = [model.model_id for model in raw_models if model.model_id not in successful]
    if failed_ids or missing:
        affected = sorted(set(failed_ids) | set(missing))
        reasons = {
            row["model_id"]: row["message"]
            for row in read_jsonl(failures)
            if row.get("stage") == "clean" and row.get("model_id") in affected
        }
        details = "; ".join(
            f"{model_id}: {reasons.get(model_id, 'No cleaned output was produced')}"
            for model_id in affected[:5]
        )
        message = (
            f"{len(affected)} of {len(raw_models)} model(s) failed or have no output. "
            f"{details}. See {failures} for details."
        )
        remaining = [model for model in raw_models if model.model_id in successful]
        if strict_clean or not remaining:
            raise CleaningError(f"Cleaning incomplete: {message}")
        print(
            f"af2rank-pipeline: warning: Skipped {len(affected)} of {len(raw_models)} input model(s). "
            f"{details}. See {failures} for details. "
            f"Continuing with {len(remaining)} cleaned model(s).",
            file=sys.stderr,
        )
    return [successful[model.model_id] for model in raw_models if model.model_id in successful]


def clean_models(
    target_spec: TargetSpec,
    models_path: str | Path,
    out_dir: str | Path,
    *,
    resume: bool = False,
    strict_clean: bool = False,
    min_identity: float = 0.90,
    min_raw_coverage: float = 0.85,
    min_target_coverage: float = 0.50,
) -> list[dict]:
    dirs = ensure_run_dirs(out_dir)
    target_spec.write_json(Path(out_dir) / "target_spec.json")

    raw_models = discover_models(models_path)
    raw_manifest = dirs["manifests"] / "raw_models.csv"
    write_raw_models_manifest(raw_models, raw_manifest)

    cleaning_manifest = dirs["manifests"] / "cleaning.jsonl"
    failures = dirs["logs"] / "failures.jsonl"
    completed = completed_ids_from_jsonl(cleaning_manifest, stage="clean") if resume else set()
    records: list[dict] = []
    failed_ids: list[str] = []

    def process_structure(raw_model: RawModel, structure: StructureRecord | None = None) -> None:
        output_pdb = dirs["cleaned"] / target_spec.target / f"{raw_model.model_id}.pdb"
        if raw_model.model_id in completed and output_pdb.exists():
            return
        try:
            if structure is None:
                record = clean_model(
                    raw_model,
                    target_spec,
                    output_pdb,
                    min_identity=min_identity,
                    min_raw_coverage=min_raw_coverage,
                    min_target_coverage=min_target_coverage,
                )
            else:
                record = _clean_structure(
                    raw_model,
                    structure,
                    target_spec,
                    output_pdb,
                    min_identity=min_identity,
                    min_raw_coverage=min_raw_coverage,
                    min_target_coverage=min_target_coverage,
                )
            append_jsonl(cleaning_manifest, record)
            records.append(record)
        except Exception as exc:
            failed_ids.append(raw_model.model_id)
            write_failure(
                failures,
                "clean",
                raw_model.model_id,
                raw_model.source_path,
                exc,
                "Inspect chain sequences, FASTA order, and whether the model is complete enough to map.",
            )

    multi_model_by_path: dict[str, dict[int, RawModel]] = {}
    for raw_model in raw_models:
        if raw_model.model_number is not None and model_suffix(raw_model.source_path) == ".pdb":
            multi_model_by_path.setdefault(raw_model.source_path, {})[raw_model.model_number] = raw_model

    streamed: set[tuple[str, int]] = set()
    for source_path, by_number in multi_model_by_path.items():
        remaining = set(by_number)
        for model_number, lines in iter_pdb_model_lines(source_path):
            if model_number is None:
                continue
            raw_model = by_number.get(model_number)
            if raw_model is None:
                continue
            streamed.add((source_path, model_number))
            remaining.discard(model_number)
            process_structure(raw_model, parse_pdb_lines(lines))
            if not remaining:
                break
        for model_number, raw_model in by_number.items():
            if (source_path, model_number) not in streamed:
                failed_ids.append(raw_model.model_id)
                write_failure(
                    failures,
                    "clean",
                    raw_model.model_id,
                    raw_model.source_path,
                    CleaningError(f"MODEL {model_number} was not found in source PDB"),
                    "Inspect whether the multi-model PDB has missing or malformed MODEL records.",
                )

    for raw_model in raw_models:
        if raw_model.model_number is not None and raw_model.source_path in multi_model_by_path:
            continue
        process_structure(raw_model)

    all_ok_records = _complete_clean_records(
        raw_models, cleaning_manifest, failed_ids, failures, strict_clean=strict_clean
    )
    decoy_list = Path(out_dir) / "cleaned" / "af2rank_decoy_list.txt"
    with decoy_list.open("w") as handle:
        for record in sorted(all_ok_records, key=lambda item: item["model_id"]):
            handle.write(f"{target_spec.target} {record['model_id']}.pdb\n")

    return all_ok_records


def clean_batch(
    target_spec: TargetSpec,
    batch_manifest: str | Path,
    out_dir: str | Path,
    *,
    run_label: str,
    resume: bool = False,
    strict_clean: bool = False,
    min_identity: float = 0.90,
    min_raw_coverage: float = 0.85,
    min_target_coverage: float = 0.50,
) -> list[dict]:
    dirs = ensure_run_dirs(out_dir)
    target_spec.write_json(Path(out_dir) / "target_spec.json")
    raw_models = read_raw_batch_manifest(batch_manifest)
    cleaning_manifest = dirs["manifests"] / f"cleaning_{run_label}.jsonl"
    failures = dirs["logs"] / f"failures_clean_{run_label}.jsonl"
    completed = completed_ids_from_jsonl(cleaning_manifest, stage="clean") if resume else set()
    records: list[dict] = []
    failed_ids: list[str] = []

    def process_structure(raw_model: RawModel, structure: StructureRecord | None = None) -> None:
        output_pdb = dirs["cleaned"] / target_spec.target / f"{raw_model.model_id}.pdb"
        if raw_model.model_id in completed and output_pdb.exists():
            return
        try:
            if structure is None:
                record = clean_model(
                    raw_model,
                    target_spec,
                    output_pdb,
                    min_identity=min_identity,
                    min_raw_coverage=min_raw_coverage,
                    min_target_coverage=min_target_coverage,
                )
            else:
                record = _clean_structure(
                    raw_model,
                    structure,
                    target_spec,
                    output_pdb,
                    min_identity=min_identity,
                    min_raw_coverage=min_raw_coverage,
                    min_target_coverage=min_target_coverage,
                )
            append_jsonl(cleaning_manifest, record)
            records.append(record)
        except Exception as exc:
            failed_ids.append(raw_model.model_id)
            write_failure(
                failures,
                "clean",
                raw_model.model_id,
                raw_model.source_path,
                exc,
                "Inspect chain sequences, FASTA order, and whether the model is complete enough to map.",
            )

    multi_model_by_path: dict[str, dict[int, RawModel]] = {}
    for raw_model in raw_models:
        if raw_model.model_number is not None and model_suffix(raw_model.source_path) == ".pdb":
            multi_model_by_path.setdefault(raw_model.source_path, {})[raw_model.model_number] = raw_model

    streamed: set[tuple[str, int]] = set()
    for source_path, by_number in multi_model_by_path.items():
        remaining = set(by_number)
        for model_number, lines in iter_pdb_model_lines(source_path):
            if model_number is None:
                continue
            raw_model = by_number.get(model_number)
            if raw_model is None:
                continue
            streamed.add((source_path, model_number))
            remaining.discard(model_number)
            process_structure(raw_model, parse_pdb_lines(lines))
            if not remaining:
                break
        for model_number, raw_model in by_number.items():
            if (source_path, model_number) not in streamed:
                failed_ids.append(raw_model.model_id)
                write_failure(
                    failures,
                    "clean",
                    raw_model.model_id,
                    raw_model.source_path,
                    CleaningError(f"MODEL {model_number} was not found in source PDB"),
                    "Inspect whether the multi-model PDB has missing or malformed MODEL records.",
                )

    for raw_model in raw_models:
        if raw_model.model_number is not None and raw_model.source_path in multi_model_by_path:
            continue
        process_structure(raw_model)
    return _complete_clean_records(
        raw_models, cleaning_manifest, failed_ids, failures, strict_clean=strict_clean
    )


def merge_clean_batch_manifests(
    out_dir: str | Path, target_prefix: str | None = None, *, strict_clean: bool = False
) -> list[dict]:
    dirs = ensure_run_dirs(out_dir)
    target_spec = load_target_spec_json(Path(out_dir) / "target_spec.json")
    if target_prefix:
        pattern = f"cleaning_{target_prefix}_batch_*.jsonl"
        failure_pattern = f"failures_clean_{target_prefix}_batch_*.jsonl"
    else:
        pattern = "cleaning_*_batch_*.jsonl"
        failure_pattern = "failures_clean_*_batch_*.jsonl"

    rows: list[dict] = []
    seen = set()
    for path in sorted(dirs["manifests"].glob(pattern)):
        for row in read_jsonl(path):
            model_id = row.get("model_id")
            if row.get("stage") != "clean" or row.get("status") != "ok" or not model_id:
                continue
            if model_id in seen:
                continue
            seen.add(model_id)
            rows.append(row)
    rows.sort(key=lambda row: row["model_id"])
    write_jsonl(dirs["manifests"] / "cleaning.jsonl", rows)

    failures = []
    for path in sorted(dirs["logs"].glob(failure_pattern)):
        failures.extend(row for row in read_jsonl(path) if row.get("model_id") not in seen)
    write_jsonl(dirs["logs"] / "failures.jsonl", failures)

    raw_manifest = dirs["manifests"] / "raw_models.csv"
    if raw_manifest.is_file():
        rows = _complete_clean_records(
            read_raw_models_manifest(raw_manifest), dirs["manifests"] / "cleaning.jsonl",
            [], dirs["logs"] / "failures.jsonl", strict_clean=strict_clean,
        )
    if not rows:
        raise CleaningError("No cleaned models found to merge")

    decoy_list = Path(out_dir) / "cleaned" / "af2rank_decoy_list.txt"
    decoy_list.parent.mkdir(parents=True, exist_ok=True)
    with decoy_list.open("w") as handle:
        for record in rows:
            handle.write(f"{target_spec.target} {record['model_id']}.pdb\n")
    return rows
