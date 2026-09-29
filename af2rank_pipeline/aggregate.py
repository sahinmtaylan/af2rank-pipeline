from __future__ import annotations

import gzip
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .manifests import ensure_run_dirs, read_csv_rows, read_jsonl, write_csv

SUMMARY_FIELDS = [
    "model_id",
    "cleaned_path",
    "output_pdb",
    "pae_json",
    "raw_npz",
    "plddt",
    "ptm",
    "i_ptm",
    "tm_i",
    "tm_o",
    "tm_io",
    "composite",
    "pae",
    "actifptm",
    "actifptm_prob",
    "actifptm_binary",
    "dockq_io",
    "best_interface",
    "best_interface_dockq",
    "mean_interface_dockq",
    "n_interfaces",
    "pae_contacts_mean",
    "pae_contacts_min",
    "pae_contacts_max",
    "pae_contacts_count",
    "missing_residue_count",
    "skipped_mismatch_count",
]

CLEAN_PREFIX_FIELDS = [
    "source_path",
    "source_format",
    "capri_md5",
    "cleaned_residue_count",
    "missing_residue_count",
    "skipped_mismatch_count",
    "chain_alignment_methods",
    "chain_raw_coverages",
    "chain_target_coverages",
    "chain_target_starts",
    "chain_target_ends",
    "chain_n_terminal_missing",
    "chain_c_terminal_missing",
]

DOCKQ_NUMERIC_FIELDS = [
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
]

PAE_SUMMARY_FIELDS = [
    "model_id",
    "interface",
    "direction",
    "pae_count",
    "pae_mean",
    "pae_min",
    "pae_max",
]


def _float_or_none(value: object) -> float | None:
    if value in (None, "", "N/A", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_fieldnames(preferred: list[str], rows: list[dict]) -> list[str]:
    fieldnames = list(preferred)
    seen = set(fieldnames)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def _stem_model_id(value: str) -> str:
    return Path(value).stem if value else ""


def _read_pae_csv(path: Path):
    if not path.is_file():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        import csv

        yield from csv.DictReader(handle)


def summarize_pae_metrics(out_dir: str | Path) -> dict[str, dict]:
    dirs = ensure_run_dirs(out_dir)
    metrics_dir = dirs["root"] / "additional_metrics"
    pae_path = metrics_dir / "pae_raw.csv.gz"
    if not pae_path.is_file():
        pae_path = metrics_dir / "pae_raw.csv"

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in _read_pae_csv(pae_path) or []:
        value = _float_or_none(row.get("pae"))
        model_id = _stem_model_id(row.get("model_id", ""))
        if value is None or not model_id:
            continue
        key = (model_id, row.get("interface", ""), row.get("direction", ""))
        grouped[key].append(value)

    rows: list[dict] = []
    by_model_values: dict[str, list[float]] = defaultdict(list)
    for (model_id, interface, direction), values in sorted(grouped.items()):
        by_model_values[model_id].extend(values)
        rows.append(
            {
                "model_id": model_id,
                "interface": interface,
                "direction": direction,
                "pae_count": len(values),
                "pae_mean": f"{mean(values):.4f}",
                "pae_min": f"{min(values):.4f}",
                "pae_max": f"{max(values):.4f}",
            }
        )

    write_csv(metrics_dir / "pae_summary.csv", rows, PAE_SUMMARY_FIELDS)
    return {
        model_id: {
            "pae_contacts_count": len(values),
            "pae_contacts_mean": f"{mean(values):.4f}",
            "pae_contacts_min": f"{min(values):.4f}",
            "pae_contacts_max": f"{max(values):.4f}",
        }
        for model_id, values in by_model_values.items()
        if values
    }


def summarize_dockq_metrics(out_dir: str | Path) -> dict[str, dict]:
    dirs = ensure_run_dirs(out_dir)
    dockq_rows = read_csv_rows(dirs["root"] / "additional_metrics" / "dockq_metrics.csv")
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in dockq_rows:
        model_id = _stem_model_id(row.get("model_id", ""))
        if model_id:
            by_model[model_id].append(row)

    summaries: dict[str, dict] = {}
    for model_id, rows in by_model.items():
        dockq_values = [
            value for value in (_float_or_none(row.get("DockQ")) for row in rows) if value is not None
        ]
        best_row = max(
            rows,
            key=lambda row: (
                value if (value := _float_or_none(row.get("DockQ"))) is not None else float("-inf")
            ),
        )
        summary = {
            "dockq_io": best_row.get("GlobalDockQ") or best_row.get("best_dockq") or "",
            "best_interface": best_row.get("interface", ""),
            "best_interface_dockq": best_row.get("DockQ", ""),
            "mean_interface_dockq": f"{mean(dockq_values):.4f}" if dockq_values else "",
            "n_interfaces": len(rows),
        }
        for key in DOCKQ_NUMERIC_FIELDS:
            summary[f"best_{key}"] = best_row.get(key, "")
        summaries[model_id] = summary
    return summaries


def aggregate_results(
    out_dir: str | Path,
    *,
    require_af2rank: bool = True,
    require_dockq: bool = True,
    allow_partial: bool = False,
) -> list[dict]:
    dirs = ensure_run_dirs(out_dir)
    expected_ids: set[str] | None = None
    if not allow_partial:
        raw_ids = {
            row["model_id"]
            for row in read_csv_rows(dirs["manifests"] / "raw_models.csv")
            if row.get("model_id")
        }
        clean_ids = {
            row["model_id"]
            for row in read_jsonl(dirs["manifests"] / "cleaning.jsonl")
            if row.get("status") == "ok" and row.get("model_id")
        }
        af2rank_ids = {
            row["model_id"]
            for row in read_csv_rows(dirs["manifests"] / "af2rank.csv")
            if row.get("status") == "ok" and row.get("model_id")
        }
        dockq_ids = {
            row["model_id"]
            for row in read_jsonl(dirs["manifests"] / "dockq_io.jsonl")
            if row.get("status") == "ok" and row.get("model_id")
        }
        expected_ids = raw_ids or clean_ids
        if not expected_ids:
            raise ValueError("No cleaned models found to aggregate")
        stages = [("cleaning", clean_ids)]
        if require_af2rank:
            stages.append(("AF2Rank", af2rank_ids))
        if require_dockq:
            stages.append(("DockQ", dockq_ids))
        for stage, observed in stages:
            missing = sorted(expected_ids - observed)
            extra = sorted(observed - expected_ids)
            if missing or extra:
                raise ValueError(
                    f"{stage} results are incomplete: {len(missing)} missing, {len(extra)} unexpected "
                    f"model(s); missing={missing[:5]}, unexpected={extra[:5]}. "
                    "Check failure logs or use allow_partial=True for exploratory analysis."
                )
    cleaning = {
        row["model_id"]: row
        for row in read_jsonl(dirs["manifests"] / "cleaning.jsonl")
        if row.get("status") == "ok"
    }
    af2rank = {
        row["model_id"]: row
        for row in read_csv_rows(dirs["manifests"] / "af2rank.csv")
        if row.get("status") == "ok"
    } if require_af2rank else {}
    dockq_manifest = {
        row["model_id"]: row
        for row in read_jsonl(dirs["manifests"] / "dockq_io.jsonl")
        if row.get("status") == "ok"
    } if require_dockq else {}
    dockq_summary = summarize_dockq_metrics(out_dir) if require_dockq else {}
    pae_summary = summarize_pae_metrics(out_dir) if require_dockq else {}

    rows: list[dict] = []
    model_ids = (
        expected_ids
        if expected_ids is not None
        else set(cleaning) | set(af2rank) | set(dockq_manifest) | set(dockq_summary)
    )
    for model_id in sorted(model_ids):
        clean_row = cleaning.get(model_id, {})
        af2_row = af2rank.get(model_id, {})
        dockq_row = {**dockq_manifest.get(model_id, {}), **dockq_summary.get(model_id, {})}
        pae_row = pae_summary.get(model_id, {})

        row = {
            "model_id": model_id,
            "cleaned_path": clean_row.get("cleaned_path", ""),
            "output_pdb": af2_row.get("output_pdb", ""),
            "pae_json": af2_row.get("pae_json", ""),
            "raw_npz": af2_row.get("raw_npz", ""),
            "missing_residue_count": clean_row.get("missing_residue_count", ""),
            "skipped_mismatch_count": clean_row.get("skipped_mismatch_count", ""),
        }
        for key, value in clean_row.items():
            if key in CLEAN_PREFIX_FIELDS:
                row[f"clean_{key}"] = value
        for key, value in af2_row.items():
            if key not in {"model_id", "status"}:
                row.setdefault(key, value)
        for key, value in dockq_row.items():
            if key not in {"model_id", "stage", "status"}:
                row[key] = value
        row.update(pae_row)
        rows.append(row)

    fieldnames = _collect_fieldnames(SUMMARY_FIELDS, rows)
    write_csv(dirs["manifests"] / "final_models.csv", rows, fieldnames)
    return rows
