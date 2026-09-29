import csv
from pathlib import Path

import pytest

from af2rank_pipeline.batching import materialize_cleaned_batch, prepare_raw_batches


def write_pdb(path: Path) -> None:
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_prepare_raw_batches_writes_balanced_csv_manifests(tmp_path: Path):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">target subunit 1\nA\n")
    models = tmp_path / "models"
    batch_root = tmp_path / "batches"
    out = tmp_path / "run"
    models.mkdir()
    for idx in range(5):
        write_pdb(models / f"model_{idx}.pdb")

    count = prepare_raw_batches("T", fasta, models, out, batch_root, num_batches=2)

    assert count == 2
    assert sorted(path.name for path in batch_root.iterdir()) == ["batch_000.csv", "batch_001.csv"]
    assert [row["model_id"] for row in read_rows(batch_root / "batch_000.csv")] == [
        "model_0",
        "model_1",
        "model_2",
    ]
    assert [row["model_id"] for row in read_rows(batch_root / "batch_001.csv")] == [
        "model_3",
        "model_4",
    ]
    assert len(read_rows(out / "manifests" / "raw_models.csv")) == 5
    assert (out / "target_spec.json").exists()


def test_prepare_raw_batches_rejects_more_batches_than_models(tmp_path: Path):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">target subunit 1\nA\n")
    models = tmp_path / "models"
    models.mkdir()
    write_pdb(models / "model_0.pdb")

    with pytest.raises(ValueError, match="cannot exceed model count"):
        prepare_raw_batches("T", fasta, models, tmp_path / "run", tmp_path / "batches", num_batches=2)


def test_materialize_batch_fails_before_copying_if_cleaned_model_is_missing(tmp_path: Path):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">A\nA\n")
    models = tmp_path / "models"
    models.mkdir()
    write_pdb(models / "m1.pdb")
    write_pdb(models / "m2.pdb")
    out = tmp_path / "run"
    batches = tmp_path / "batches"
    prepare_raw_batches("T", fasta, models, out, batches, num_batches=1)
    cleaned = out / "cleaned" / "af2rank_input" / "T"
    cleaned.mkdir(parents=True)
    write_pdb(cleaned / "m1.pdb")
    dest = tmp_path / "materialized"

    with pytest.raises(FileNotFoundError, match="m2"):
        materialize_cleaned_batch("T", out, batches / "batch_000.csv", dest)
    assert not dest.exists()
