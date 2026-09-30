import ast
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from af2rank_pipeline.aggregate import aggregate_results
from af2rank_pipeline.batching import materialize_cleaned_batch, write_raw_batch_manifest
from af2rank_pipeline.cleaning import clean_batch, clean_models, merge_clean_batch_manifests
from af2rank_pipeline.cli import main
from af2rank_pipeline.exceptions import CleaningError
from af2rank_pipeline.ingest import discover_models
from af2rank_pipeline.manifests import read_csv_rows, read_jsonl, write_csv
from af2rank_pipeline.target import load_target_spec

ATOM = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"


def mixed_inputs(tmp_path):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">A\nA\n")
    models = tmp_path / "models"
    models.mkdir()
    (models / "good.pdb").write_text(ATOM + "END\n")
    (models / "bad.pdb").write_text("END\n")
    return fasta, models


@pytest.mark.parametrize("strict", [False, True])
def test_cleaning_policy_and_resume(tmp_path, capsys, strict):
    fasta, models = mixed_inputs(tmp_path)
    spec = load_target_spec("T", fasta)
    out = tmp_path / "out"
    for resume in [False, True]:
        if strict:
            with pytest.raises(CleaningError, match="bad: No protein chains found"):
                clean_models(spec, models, out, strict_clean=True, resume=resume)
        else:
            records = clean_models(spec, models, out, resume=resume)
            assert [r["model_id"] for r in records] == ["good"]
            warning = capsys.readouterr().err
            assert "Skipped 1 of 2" in warning
            assert "bad: No protein chains found" in warning
            assert "Continuing with 1 cleaned model" in warning
            assert (out / "cleaned/af2rank_decoy_list.txt").read_text() == "T good.pdb\n"
        assert (out / "cleaned/af2rank_input/T/good.pdb").is_file()
        assert read_jsonl(out / "logs/failures.jsonl")[-1]["model_id"] == "bad"


def test_default_full_run_scores_only_surviving_models(tmp_path, monkeypatch, capsys):
    fasta, models = mixed_inputs(tmp_path)
    out = tmp_path / "out"

    def score(spec, run_out, config, **kwargs):
        run_out = Path(run_out)
        assert sorted(p.stem for p in (run_out / "cleaned/af2rank_input/T").glob("*.pdb")) == ["good"]
        records = [{"model_id": "good", "status": "ok", "plddt": "80"}]
        write_csv(run_out / "manifests/af2rank.csv", records, ["model_id", "status", "plddt"])
        return records

    inference = Mock(side_effect=score)
    monkeypatch.setattr("af2rank_pipeline.af2rank_runner.run_af2rank_stage", inference)
    command = ["run-target", "--target", "T", "--fasta", str(fasta), "--models", str(models),
               "--out", str(out), "--params", "unused", "--skip-dockq"]
    assert main(command) == 0
    assert [r["model_id"] for r in read_csv_rows(out / "manifests/final_models.csv")] == ["good"]
    assert "Skipped 1 of 2" in capsys.readouterr().err
    inference.reset_mock()
    assert main(command + ["--strict-clean"]) == 1
    inference.assert_not_called()
    assert "Cleaning incomplete" in capsys.readouterr().err


def test_aggregation_requires_all_cleaned_scores_even_when_raw_models_failed(tmp_path):
    fasta, models = mixed_inputs(tmp_path)
    out = tmp_path / "out"
    clean_models(load_target_spec("T", fasta), models, out)
    write_csv(out / "manifests/af2rank.csv", [{"model_id": "good", "status": "ok"}], ["model_id", "status"])
    assert [r["model_id"] for r in aggregate_results(out, require_dockq=False)] == ["good"]
    with pytest.raises(ValueError, match="cleaning results are incomplete"):
        aggregate_results(out, strict_clean=True, require_dockq=False)
    write_csv(out / "manifests/af2rank.csv", [], ["model_id", "status"])
    with pytest.raises(ValueError, match="AF2Rank results are incomplete"):
        aggregate_results(out, require_dockq=False)
    write_csv(out / "manifests/af2rank.csv", [{"model_id": "good", "status": "ok"}], ["model_id", "status"])
    with pytest.raises(ValueError, match="DockQ results are incomplete"):
        aggregate_results(out)


def test_batch_clean_merge_and_materialize_follow_cleaning_policy(tmp_path):
    fasta, models = mixed_inputs(tmp_path)
    spec = load_target_spec("T", fasta)
    raw = discover_models(models)
    batches = tmp_path / "batches"
    batch = write_raw_batch_manifest(batches, 0, raw)
    out = tmp_path / "out"
    records = clean_batch(spec, batch, out, run_label="T_batch_000")
    assert [r["model_id"] for r in records] == ["good"]
    write_csv(out / "manifests/raw_models.csv", [r.to_row() for r in raw], list(raw[0].to_row()))
    assert [r["model_id"] for r in merge_clean_batch_manifests(out)] == ["good"]
    with pytest.raises(CleaningError, match="Cleaning incomplete"):
        merge_clean_batch_manifests(out, strict_clean=True)
    dest = tmp_path / "materialized"
    assert materialize_cleaned_batch("T", out, batch, dest) == 1
    assert [p.name for p in dest.iterdir()] == ["good.pdb"]
    with pytest.raises(FileNotFoundError, match="bad"):
        materialize_cleaned_batch("T", out, batch, tmp_path / "strict", strict_clean=True)
    with pytest.raises(CleaningError, match="Cleaning incomplete"):
        clean_batch(spec, batch, out, run_label="T_batch_000", strict_clean=True)


def test_notebook_explicitly_requests_strict_clean():
    repo = Path(__file__).resolve().parents[1]
    notebook = json.loads((repo / "notebooks/colab_demo.ipynb").read_text())
    cell = next(c for c in notebook["cells"] if c["metadata"].get("id") == "run")
    node = next(n for n in ast.parse("".join(cell["source"])).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "command" for t in n.targets))
    assert any(isinstance(arg, ast.Constant) and arg.value == "--strict-clean" for arg in node.value.elts)
    assert cell["metadata"].get("cellView") != "form"
