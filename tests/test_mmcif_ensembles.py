from pathlib import Path

import pytest

from af2rank_pipeline.batching import write_raw_batch_manifest
from af2rank_pipeline.cleaning import clean_batch, clean_models
from af2rank_pipeline.exceptions import PipelineError
from af2rank_pipeline.ingest import discover_models, read_raw_models_manifest
from af2rank_pipeline.structure import parse_mmcif
from af2rank_pipeline.target import load_target_spec


def write_ensemble(path: Path, numbers: list[int]) -> None:
    columns = [
        "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
        "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
        "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
        "B_iso_or_equiv", "auth_seq_id", "auth_comp_id", "auth_asym_id",
        "auth_atom_id", "pdbx_PDB_model_num",
    ]
    lines = ["data_ensemble\n#\nloop_\n"] + [f"_atom_site.{c}\n" for c in columns]
    atom_id = 0
    for index, number in enumerate(numbers):
        for residue, name in [(1, "ALA"), (2, "CYS")]:
            atom_id += 1
            x = index * 10 + residue
            lines.append(
                f"ATOM {atom_id} C CA . {name} A 1 {residue} ? "
                f"{x}.0 0.0 0.0 1.0 80.0 {residue} {name} A CA {number}\n"
            )
    path.write_text("".join(lines) + "#\n")


@pytest.mark.parametrize("suffix", [".cif", ".mmcif"])
@pytest.mark.parametrize("numbers", [[1, 2], [7, 42], [42, 7]])
def test_discovers_and_cleans_each_mmcif_member(tmp_path, suffix, numbers):
    path = tmp_path / f"ensemble{suffix}"
    write_ensemble(path, numbers)
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">A\nAC\n")
    models = discover_models(path)
    assert [m.model_number for m in models] == numbers
    assert [m.model_id for m in models] == [f"ensemble_model_{n:04d}" for n in numbers]
    assert all(m.source_path == str(path) and m.capri_md5 is None for m in models)

    out = tmp_path / "out"
    records = clean_models(load_target_spec("T", fasta), path, out, strict_clean=True)
    assert [r["model_number"] for r in records] == numbers
    assert [r.model_number for r in read_raw_models_manifest(out / "manifests/raw_models.csv")] == numbers
    for index, record in enumerate(records):
        assert record["cleaned_residue_count"] == 2 and record["missing_residue_count"] == 0
        cleaned = Path(record["cleaned_path"])
        atoms = [line for line in cleaned.read_text().splitlines() if line.startswith("ATOM")]
        assert [float(line[30:38]) for line in atoms] == [index * 10 + 1, index * 10 + 2]
    assert (out / "cleaned/af2rank_decoy_list.txt").read_text().splitlines() == [
        f"T ensemble_model_{n:04d}.pdb" for n in sorted(numbers)
    ]


def test_single_model_keeps_filename_id_and_selects_first_by_default(tmp_path):
    path = tmp_path / "single.cif"
    write_ensemble(path, [7])
    models = discover_models(path)
    assert len(models) == 1
    assert models[0].model_id == "single" and models[0].model_number is None
    assert parse_mmcif(path).chains[0].sequence == "AC"
    assert parse_mmcif(path, model_number=7).chains[0].sequence == "AC"


def test_model_number_is_literal_not_position(tmp_path):
    path = tmp_path / "ensemble.cif"
    write_ensemble(path, [7, 42])
    assert parse_mmcif(path).chains[0].residues[0].atoms[0].x == 1
    assert parse_mmcif(path, model_number=42).chains[0].residues[0].atoms[0].x == 11
    with pytest.raises(PipelineError, match="Model 2 not found"):
        parse_mmcif(path, model_number=2)


def test_mmcif_ensemble_survives_batch_discovery_and_cleaning(tmp_path):
    path = tmp_path / "ensemble.cif"
    write_ensemble(path, [3, 9])
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">A\nAC\n")
    batch = write_raw_batch_manifest(tmp_path / "batches", 0, discover_models(path))
    records = clean_batch(load_target_spec("T", fasta), batch, tmp_path / "out",
                          run_label="T_batch_000", strict_clean=True)
    assert [r["model_number"] for r in records] == [3, 9]
    assert len({Path(r["cleaned_path"]).read_text() for r in records}) == 2
