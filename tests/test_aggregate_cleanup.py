import gzip
import json
from pathlib import Path

from af2rank_pipeline.aggregate import aggregate_results, summarize_dockq_metrics
from af2rank_pipeline.cleanup import cleanup_outputs
from af2rank_pipeline.dockq_pae import load_pae_matrix
from af2rank_pipeline.manifests import append_jsonl, read_csv_rows, write_csv


def test_aggregate_writes_final_models_and_pae_summary(tmp_path: Path):
    out = tmp_path / "run"
    manifests = out / "manifests"
    metrics = out / "additional_metrics"

    append_jsonl(
        manifests / "cleaning.jsonl",
        {
            "stage": "clean",
            "status": "ok",
            "model_id": "m1",
            "cleaned_path": "cleaned/m1.pdb",
            "missing_residue_count": 0,
            "skipped_mismatch_count": 0,
        },
    )
    write_csv(
        manifests / "af2rank.csv",
        [{"model_id": "m1", "status": "ok", "plddt": "90", "ptm": "0.5", "actifptm_A_B": "0.8"}],
        ["model_id", "status", "plddt", "ptm", "actifptm_A_B"],
    )
    append_jsonl(
        manifests / "dockq_io.jsonl",
        {"stage": "dockq_io", "status": "ok", "model_id": "m1", "dockq_io": "0.4"},
    )
    write_csv(
        metrics / "dockq_metrics.csv",
        [
            {"model_id": "m1.pdb", "interface": "AB", "DockQ": "0.25", "GlobalDockQ": "0.4"},
            {"model_id": "m1.pdb", "interface": "AC", "DockQ": "0.75", "GlobalDockQ": "0.4"},
            {"model_id": "old.pdb", "interface": "AB", "DockQ": "0.99", "GlobalDockQ": "0.99"},
        ],
        ["model_id", "interface", "DockQ", "GlobalDockQ"],
    )
    metrics.mkdir(parents=True, exist_ok=True)
    with gzip.open(metrics / "pae_raw.csv.gz", "wt", newline="") as handle:
        handle.write("model_id,interface,direction,pae\n")
        handle.write("m1.pdb,AB,A-B,2.0\n")
        handle.write("m1.pdb,AB,A-B,4.0\n")

    rows = aggregate_results(out)

    assert [row["model_id"] for row in rows] == ["m1"]
    assert rows[0]["best_interface"] == "AC"
    assert rows[0]["best_interface_dockq"] == "0.75"
    assert rows[0]["pae_contacts_mean"] == "3.0000"
    assert read_csv_rows(manifests / "final_models.csv")[0]["actifptm_A_B"] == "0.8"
    assert not (manifests / "final_scores.csv").exists()
    assert read_csv_rows(metrics / "pae_summary.csv")[0]["pae_mean"] == "3.0000"


def test_cleanup_removes_raw_npz_and_keeps_scored_pdbs(tmp_path: Path):
    out = tmp_path / "run"
    raw_npz = out / "af2rank" / "batch_000" / "alphafold_ptm_model2_rec1" / "raw_npz"
    scored = out / "af2rank" / "batch_000" / "alphafold_ptm_model2_rec1" / "scored_pdbs"
    pae = out / "af2rank" / "pae"
    raw_npz.mkdir(parents=True)
    scored.mkdir(parents=True)
    pae.mkdir(parents=True)
    (raw_npz / "m1_raw_outputs.npz").write_text("raw")
    (scored / "m1_scored.pdb").write_text("pdb")
    (pae / "m1.json").write_text("{}")

    summary = cleanup_outputs(out)

    assert not raw_npz.exists()
    assert scored.exists()
    assert not (pae / "m1.json").exists()
    assert (pae / "m1.json.gz").exists()
    assert summary["archive_written"] is False
    assert summary["removed"] == [str(raw_npz)]


def test_zero_dockq_is_ranked_above_missing_value(tmp_path: Path):
    write_csv(
        tmp_path / "additional_metrics" / "dockq_metrics.csv",
        [
            {"model_id": "m1.pdb", "interface": "AB", "DockQ": ""},
            {"model_id": "m1.pdb", "interface": "AC", "DockQ": "0.0"},
        ],
        ["model_id", "interface", "DockQ"],
    )

    summary = summarize_dockq_metrics(tmp_path)["m1"]
    assert summary["best_interface"] == "AC"
    assert summary["best_interface_dockq"] == "0.0"


def test_load_pae_matrix_reads_gzipped_json(tmp_path: Path):
    path = tmp_path / "pae.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump({"pae": [[0, 1], [2, 3]]}, handle)

    assert load_pae_matrix(path).shape == (2, 2)
