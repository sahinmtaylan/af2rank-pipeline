from pathlib import Path

import pytest

from af2rank_pipeline.aggregate import aggregate_results
from af2rank_pipeline.cli import build_parser, main
from af2rank_pipeline.actifptm_runner import parse_args as parse_actifptm_args
from af2rank_pipeline.exceptions import CleaningError
from af2rank_pipeline.manifests import append_jsonl, read_csv_rows, read_jsonl, write_csv
from af2rank_pipeline.cleaning import clean_models
from af2rank_pipeline.target import load_target_spec


def test_clean_command_fails_when_no_input_model_can_be_cleaned(tmp_path: Path, capsys):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">chain_A\nAC\n")
    bad_pdb = tmp_path / "bad.pdb"
    bad_pdb.write_text("END\n")
    out = tmp_path / "run"

    assert main([
        "clean", "--target", "example", "--fasta", str(fasta),
        "--models", str(bad_pdb), "--out", str(out),
    ]) == 1
    error = capsys.readouterr().err
    assert "af2rank-pipeline: error: Cleaning incomplete: 1 of 1" in error
    assert "bad: No protein chains found in model" in error
    assert "Traceback" not in error

    assert (out / "logs" / "failures.jsonl").is_file()


def test_aggregate_rejects_partial_scores_unless_requested(tmp_path: Path):
    out = tmp_path / "run"
    append_jsonl(
        out / "manifests" / "cleaning.jsonl",
        {"stage": "clean", "status": "ok", "model_id": "m1"},
    )

    with pytest.raises(ValueError, match="AF2Rank results are incomplete"):
        aggregate_results(out)

    rows = aggregate_results(out, allow_partial=True)
    assert [row["model_id"] for row in rows] == ["m1"]


def test_manifest_paths_survive_a_change_of_working_directory(tmp_path: Path, monkeypatch):
    (tmp_path / "target.fasta").write_text(">A\nA\n")
    (tmp_path / "m.pdb").write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
    )
    monkeypatch.chdir(tmp_path)
    target_spec = load_target_spec("T", "target.fasta")
    clean_models(target_spec, "m.pdb", "run")
    record = read_jsonl(tmp_path / "run" / "manifests" / "cleaning.jsonl")[0]

    monkeypatch.chdir(tmp_path.parent)
    assert Path(record["source_path"]).is_file()
    assert Path(record["cleaned_path"]).is_file()
    assert Path(target_spec.fasta_path).is_file()


def test_skipping_dockq_does_not_reuse_old_metrics(tmp_path: Path):
    out = tmp_path / "run"
    append_jsonl(
        out / "manifests" / "cleaning.jsonl",
        {"stage": "clean", "status": "ok", "model_id": "m1"},
    )
    write_csv(
        out / "manifests" / "af2rank.csv",
        [{"model_id": "m1", "status": "ok"}],
        ["model_id", "status"],
    )
    write_csv(
        out / "additional_metrics" / "dockq_metrics.csv",
        [{"model_id": "m1.pdb", "DockQ": "0.9", "GlobalDockQ": "0.9"}],
        ["model_id", "DockQ", "GlobalDockQ"],
    )

    rows = aggregate_results(out, require_dockq=False)
    assert [row["model_id"] for row in rows] == ["m1"]
    assert read_csv_rows(out / "manifests" / "final_models.csv")[0]["dockq_io"] == ""


def test_zero_iterations_is_rejected_by_both_clis(monkeypatch, capsys):
    with pytest.raises(SystemExit) as pipeline_error:
        build_parser().parse_args([
            "run-target", "--target", "T", "--fasta", "target.fasta",
            "--out", "run", "--models", "models", "--params", "params",
            "--colabdesign", "ColabDesign", "--iterations", "0",
        ])
    assert pipeline_error.value.code == 2
    assert "iterations must be at least 1" in capsys.readouterr().err

    monkeypatch.setattr(
        "sys.argv",
        [
            "actifptm_runner.py", "--decoy_dir", "models", "--params", "params",
            "--colabdesign", "ColabDesign", "--output_dir", "run", "--iterations", "0",
        ],
    )
    with pytest.raises(SystemExit) as runner_error:
        parse_actifptm_args()
    assert runner_error.value.code == 2
    assert "iterations must be at least 1" in capsys.readouterr().err
