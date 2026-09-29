"""DockQ and PAE metrics used by the pipeline."""

from __future__ import annotations

import concurrent.futures
import csv
import gzip
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


PAE_COMPACT_FIELDNAMES = [
    "batch",
    "model_id",
    "interface",
    "scored_chain",
    "scored_resname",
    "scored_resnum",
    "aligned_chain",
    "aligned_resname",
    "aligned_resnum",
    "direction",
    "pae",
]

DOCKQ_BASE_FIELDNAMES = [
    "run_name",
    "batch",
    "model_id",
    "model",
    "native",
    "interface",
    "best_dockq",
    "GlobalDockQ",
    "best_mapping_str",
]

DOCKQ_INTERFACE_FIELDNAMES = [
    "DockQ",
    "F1",
    "iRMSD",
    "LRMSD",
    "fnat",
    "nat_correct",
    "nat_total",
    "fnonnat",
    "nonnat_count",
    "model_total",
    "clashes",
    "len1",
    "len2",
    "chain1",
    "chain2",
    "class1",
    "class2",
    "is_het",
]

AF2RANK_METADATA_FIELDNAMES = ["batch", "model_id", "af2rank_csv"]


@dataclass(frozen=True)
class RunSpec:
    name: str
    batch: str
    model_id: str
    native: Path
    model: Path
    pae: Path
    af2rank_csv: Path | None = None
    mapping: str | None = None
    pae_chain_order: str | None = None


@dataclass(frozen=True)
class RunOutput:
    spec: RunSpec
    pae_rows_path: Path
    pae_row_count: int
    dockq_rows: list[dict[str, object]]
    dockq_json: Path


def load_pae_matrix(path: Path | str) -> np.ndarray:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "pae" not in data:
            raise ValueError(f"{path} is a JSON object but has no top-level 'pae' key")
        data = data["pae"]
    elif not isinstance(data, list):
        raise ValueError(f"{path} must be either a raw matrix list or an object with 'pae'")

    pae = np.asarray(data, dtype=float)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"{path} contains a non-square PAE matrix: {pae.shape}")
    return pae


def parse_chain_order(chain_order: str | Iterable[str] | None) -> tuple[str, ...] | None:
    if not chain_order:
        return None
    if isinstance(chain_order, str):
        if "," in chain_order:
            chains = tuple(chain.strip() for chain in chain_order.split(",") if chain.strip())
        else:
            chains = tuple(chain_order.strip())
    else:
        chains = tuple(chain_order)
    if not chains:
        raise ValueError("PAE chain order was provided but no chain IDs were found")
    if len(set(chains)) != len(chains):
        raise ValueError(f"PAE chain order has duplicate chain IDs: {chain_order}")
    return chains


def read_pdb_residues_by_chain(pdb_file: Path | str) -> tuple[dict[str, list[int]], tuple[str, ...]]:
    residues_by_chain: dict[str, list[int]] = {}
    chain_order: list[str] = []
    seen: set[tuple[str, int, str]] = set()
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21].strip() or " "
            resseq = line[22:26].strip()
            insertion_code = line[26].strip()
            try:
                resnum = int(resseq)
            except ValueError:
                continue
            key = (chain_id, resnum, insertion_code)
            if key in seen:
                continue
            seen.add(key)
            if chain_id not in residues_by_chain:
                residues_by_chain[chain_id] = []
                chain_order.append(chain_id)
            residues_by_chain[chain_id].append(resnum)
    return residues_by_chain, tuple(chain_order)


def build_residue_to_pae_index(
    pdb_file: Path | str,
    pae_chain_order: str | Iterable[str] | None = None,
) -> dict[tuple[str, int], int]:
    residues_by_chain, pdb_chain_order = read_pdb_residues_by_chain(pdb_file)
    chain_order = parse_chain_order(pae_chain_order) or pdb_chain_order
    missing = [chain for chain in chain_order if chain not in residues_by_chain]
    if missing:
        raise ValueError(
            f"PAE chain order references chains not found in {pdb_file}: {', '.join(missing)}"
        )
    if pae_chain_order:
        unlisted = [chain for chain in pdb_chain_order if chain not in chain_order]
        if unlisted:
            raise ValueError(
                f"PAE chain order does not include all model chains from {pdb_file}: "
                f"{', '.join(unlisted)}"
            )

    mapping: dict[tuple[str, int], int] = {}
    index = 0
    for chain_id in chain_order:
        for resnum in residues_by_chain[chain_id]:
            mapping[(chain_id, resnum)] = index
            index += 1
    return mapping


def model_id_from_scored_pdb(scored_pdb: Path) -> str:
    suffix = "_scored.pdb"
    name = scored_pdb.name
    if not name.endswith(suffix):
        raise ValueError(f"{scored_pdb} does not end with {suffix}")
    return f"{name[:-len(suffix)]}.pdb"


def read_manifest(
    manifest: Path | str,
    mapping: str | None = None,
    pae_chain_order: str | None = None,
) -> list[RunSpec]:
    manifest = Path(manifest)
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        missing = {"native", "model", "pae"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{manifest} is missing columns: {', '.join(sorted(missing))}")
        specs = []
        for index, row in enumerate(reader, start=1):
            model = Path(row["model"])
            model_id = row.get("model_id") or (
                model_id_from_scored_pdb(model) if model.name.endswith("_scored.pdb") else model.name
            )
            name = row.get("name") or Path(model_id).stem or f"run_{index}"
            specs.append(
                RunSpec(
                    name=name,
                    batch=row.get("batch") or manifest.stem,
                    model_id=model_id,
                    native=Path(row["native"]),
                    model=model,
                    pae=Path(row["pae"]),
                    af2rank_csv=Path(row["af2rank_csv"]) if row.get("af2rank_csv") else None,
                    mapping=row.get("mapping") or mapping,
                    pae_chain_order=row.get("pae_chain_order") or row.get("chain_order") or pae_chain_order,
                )
            )
    return specs


def validate_run_specs(specs: Iterable[RunSpec]) -> list[str]:
    issues: list[str] = []
    for spec in specs:
        for label, path in missing_required_files(spec):
            issues.append(f"{spec.batch}/{spec.model_id}: missing {label}: {path}")
        if spec.af2rank_csv and not spec.af2rank_csv.is_file():
            issues.append(f"{spec.batch}/{spec.model_id}: missing AF2Rank CSV: {spec.af2rank_csv}")
    return issues


def missing_required_files(spec: RunSpec) -> list[tuple[str, Path]]:
    missing: list[tuple[str, Path]] = []
    for label, path in [("native", spec.native), ("model", spec.model), ("pae", spec.pae)]:
        if not path.is_file():
            missing.append((label, path))
    return missing


def run_dockq(
    spec: RunSpec,
    dockq_json: Path,
) -> Path:
    dockq_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "DockQ.DockQ",
        str(spec.model),
        str(spec.native),
        "--json",
        str(dockq_json),
        "--short",
    ]
    if spec.mapping:
        cmd.extend(["--mapping", spec.mapping])
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if len(detail) > 4000:
            detail = detail[-4000:]
        raise ValueError(
            f"DockQ failed for {spec.batch}/{spec.model_id} with exit code {result.returncode}: "
            f"{detail or 'no stdout/stderr captured'}"
        )
    return dockq_json


def extract_direction_rows(
    spec: RunSpec,
    interface: str,
    direction: str,
    scored_residues: list[dict[str, object]],
    aligned_residues: list[dict[str, object]],
    scored_chain: str,
    aligned_chain: str,
    pae_matrix: np.ndarray,
    res_to_idx: dict[tuple[str, int], int],
) -> Iterator[dict[str, object]]:
    scored_pairs = [
        (residue, res_to_idx.get((scored_chain, int(residue["resnum"]))))
        for residue in scored_residues
    ]
    aligned_pairs = [
        (residue, res_to_idx.get((aligned_chain, int(residue["resnum"]))))
        for residue in aligned_residues
    ]
    scored = [(residue, index) for residue, index in scored_pairs if index is not None]
    aligned = [(residue, index) for residue, index in aligned_pairs if index is not None]
    if not scored or not aligned:
        return

    scored_indices = [int(index) for _, index in scored]
    aligned_indices = [int(index) for _, index in aligned]
    submatrix = pae_matrix[np.ix_(scored_indices, aligned_indices)]
    for row_index, (scored_residue, _) in enumerate(scored):
        for col_index, (aligned_residue, _) in enumerate(aligned):
            yield {
                "batch": spec.batch,
                "model_id": spec.model_id,
                "interface": interface,
                "scored_chain": scored_chain,
                "scored_resname": scored_residue.get("resname"),
                "scored_resnum": scored_residue.get("resnum"),
                "aligned_chain": aligned_chain,
                "aligned_resname": aligned_residue.get("resname"),
                "aligned_resnum": aligned_residue.get("resnum"),
                "direction": direction,
                "pae": float(submatrix[row_index, col_index]),
            }


def iter_pae_rows(
    spec: RunSpec,
    dockq_data: dict[str, object],
    pae_matrix: np.ndarray,
) -> Iterator[dict[str, object]]:
    # DockQ contact/interface residue sets are emitted in native/input residue
    # numbering. The PAE matrix follows the AF2Rank input sequence order, so the
    # native PDB is the stable reference even when ColabDesign renumbers scored
    # PDB outputs.
    res_to_idx = build_residue_to_pae_index(spec.native, spec.pae_chain_order)
    if len(res_to_idx) != pae_matrix.shape[0]:
        raise ValueError(
            f"{spec.model_id}: PAE size {pae_matrix.shape[0]} does not match "
            f"{len(res_to_idx)} residues in {spec.native}"
        )

    best_result = dockq_data.get("best_result", {})
    if not isinstance(best_result, dict):
        raise ValueError(f"{spec.model_id}: DockQ JSON has no object best_result")

    for interface, raw_result in best_result.items():
        if not isinstance(raw_result, dict):
            continue
        residue_set = raw_result.get("native_contacts")
        if not isinstance(residue_set, dict):
            continue
        chain1 = str(raw_result.get("chain1", interface[0]))
        chain2 = str(raw_result.get("chain2", interface[-1]))
        chain1_residues = residue_set.get("chain1") or []
        chain2_residues = residue_set.get("chain2") or []
        if not isinstance(chain1_residues, list) or not isinstance(chain2_residues, list):
            continue

        yield from extract_direction_rows(
            spec,
            str(interface),
            f"{chain1}-{chain2}",
            chain1_residues,
            chain2_residues,
            chain1,
            chain2,
            pae_matrix,
            res_to_idx,
        )
        yield from extract_direction_rows(
            spec,
            str(interface),
            f"{chain2}-{chain1}",
            chain2_residues,
            chain1_residues,
            chain2,
            chain1,
            pae_matrix,
            res_to_idx,
        )


def is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def flatten_dockq_metrics(spec: RunSpec, dockq_data: dict[str, object]) -> list[dict[str, object]]:
    top_level = {
        "best_dockq": dockq_data.get("best_dockq"),
        "GlobalDockQ": dockq_data.get("GlobalDockQ"),
        "best_mapping_str": dockq_data.get("best_mapping_str"),
    }
    best_result = dockq_data.get("best_result", {})
    if not isinstance(best_result, dict):
        return []

    rows: list[dict[str, object]] = []
    for interface, raw_result in best_result.items():
        if not isinstance(raw_result, dict):
            continue
        row: dict[str, object] = {
            "run_name": spec.name,
            "batch": spec.batch,
            "model_id": spec.model_id,
            "model": str(spec.model),
            "native": str(spec.native),
            "interface": interface,
            **top_level,
        }
        for key, value in raw_result.items():
            if is_scalar(value):
                row[key] = value
        rows.append(row)
    return rows


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def format_pae_row(row: dict[str, object]) -> dict[str, object]:
    formatted = dict(row)
    formatted["pae"] = f"{float(row['pae']):.3f}"
    return formatted


def write_pae_rows_without_header(
    path: Path,
    rows: Iterable[dict[str, object]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAE_COMPACT_FIELDNAMES, extrasaction="ignore")
        for row in rows:
            writer.writerow(format_pae_row(row))
            count += 1
    return count


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def process_run(
    spec: RunSpec,
    outdir: Path,
) -> RunOutput:
    dockq_json = outdir / "dockq_json" / f"{safe_name(Path(spec.model_id).stem)}.json"
    run_dockq(spec, dockq_json)
    with open(dockq_json) as f:
        dockq_data = json.load(f)
    pae_matrix = load_pae_matrix(spec.pae)
    tmp_pae = outdir / ".tmp" / f"{safe_name(Path(spec.model_id).stem)}.pae_rows.csv.gz"
    pae_count = write_pae_rows_without_header(
        tmp_pae,
        iter_pae_rows(spec, dockq_data, pae_matrix),
    )
    if pae_count == 0:
        raise ValueError(
            f"{spec.model_id}: no PAE rows extracted. This pipeline expects native_contacts "
            "from the bundled modified DockQ output."
        )
    return RunOutput(
        spec=spec,
        pae_rows_path=tmp_pae,
        pae_row_count=pae_count,
        dockq_rows=flatten_dockq_metrics(spec, dockq_data),
        dockq_json=dockq_json,
    )


def collect_csv_fieldnames(
    preferred: list[str],
    rows: Iterable[dict[str, object]],
) -> list[str]:
    seen = set(preferred)
    fieldnames = list(preferred)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def write_final_pae_csv(
    path: Path,
    run_outputs: list[RunOutput],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with gzip.open(path, "wt", newline="") as out_f:
        out_f.write(",".join(PAE_COMPACT_FIELDNAMES) + "\n")
        for output in run_outputs:
            if not output.pae_rows_path.is_file():
                continue
            with gzip.open(output.pae_rows_path, "rt", newline="") as in_f:
                shutil.copyfileobj(in_f, out_f)
            total += output.pae_row_count
    return total


def collect_af2rank_rows(
    specs: Iterable[RunSpec],
) -> tuple[list[dict[str, object]], list[str]]:
    specs = list(specs)
    model_ids = {spec.model_id for spec in specs}
    csv_to_batches: dict[Path, set[str]] = {}
    for spec in specs:
        if spec.af2rank_csv:
            csv_to_batches.setdefault(spec.af2rank_csv, set()).add(spec.batch)

    rows: list[dict[str, object]] = []
    original_fieldnames: list[str] = []
    seen_original = set()
    for score_csv, batches in sorted(csv_to_batches.items()):
        if not score_csv.is_file():
            continue
        with open(score_csv, newline="") as f:
            reader = csv.DictReader(f)
            for fieldname in reader.fieldnames or []:
                if fieldname not in seen_original and fieldname not in AF2RANK_METADATA_FIELDNAMES:
                    seen_original.add(fieldname)
                    original_fieldnames.append(fieldname)
            for row in reader:
                row_model_id = row.get("id") or row.get("model_id") or ""
                if row_model_id and row_model_id not in model_ids:
                    continue
                metadata = {
                    "batch": sorted(batches)[0] if len(batches) == 1 else ";".join(sorted(batches)),
                    "model_id": row_model_id,
                    "af2rank_csv": str(score_csv),
                }
                rows.append({**row, **metadata})
    return rows, AF2RANK_METADATA_FIELDNAMES + original_fieldnames


def process_specs(specs: list[RunSpec], outdir: Path, *, jobs: int = 1) -> int:
    if jobs < 1:
        raise ValueError("--jobs must be at least 1")
    if not specs:
        raise ValueError("No DockQ runs found in manifest")
    issues = validate_run_specs(specs)
    if issues:
        raise ValueError("Missing required files:\n" + "\n".join(issues[:25]))

    outdir.mkdir(parents=True, exist_ok=True)
    if jobs == 1:
        outputs = [process_run(spec, outdir) for spec in specs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            outputs = list(executor.map(lambda spec: process_run(spec, outdir), specs))
    outputs.sort(key=lambda output: (output.spec.batch, output.spec.model_id))

    pae_output = outdir / "pae_raw.csv.gz"
    pae_count = write_final_pae_csv(pae_output, outputs)
    dockq_rows = [row for output in outputs for row in output.dockq_rows]
    dockq_fieldnames = collect_csv_fieldnames(
        DOCKQ_BASE_FIELDNAMES + DOCKQ_INTERFACE_FIELDNAMES,
        dockq_rows,
    )
    dockq_count = write_rows(outdir / "dockq_metrics.csv", dockq_fieldnames, dockq_rows)
    af2rank_rows, af2rank_fieldnames = collect_af2rank_rows(specs)
    af2rank_count = write_rows(outdir / "af2rank_metrics.csv", af2rank_fieldnames, af2rank_rows)

    for output in outputs:
        output.pae_rows_path.unlink(missing_ok=True)

    print(f"Wrote {pae_output} ({pae_count} rows)")
    print(f"Wrote {outdir / 'dockq_metrics.csv'} ({dockq_count} rows)")
    print(f"Wrote {outdir / 'af2rank_metrics.csv'} ({af2rank_count} rows)")
    print(f"Wrote {len(outputs)} DockQ JSON file(s) under {outdir / 'dockq_json'}")
    return 0
