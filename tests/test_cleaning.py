import gzip
from pathlib import Path
from unittest.mock import patch

import pytest

from af2rank_pipeline.batching import write_raw_batch_manifest
from af2rank_pipeline.cleaning import clean_batch, clean_model, clean_models, merge_clean_batch_manifests
from af2rank_pipeline.exceptions import CleaningError
from af2rank_pipeline.ingest import RawModel, discover_models
from af2rank_pipeline.manifests import read_jsonl
from af2rank_pipeline.target import load_target_spec


def atom_line(serial, atom, resname, chain, resseq, x=0.0):
    return (
        f"ATOM  {serial:5d} {atom:>4} {resname:>3} {chain:1s}{resseq:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}           {atom.strip()[0]:>2}\n"
    )


def residue(serial_start, resname, chain, resseq):
    return "".join(
        atom_line(serial_start + i, atom, resname, chain, resseq, x=float(serial_start + i))
        for i, atom in enumerate(["N", "CA", "C", "O"])
    )


def test_clean_model_reorders_and_numbers_each_chain_from_one(tmp_path: Path):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">target subunit 1\nAC\n>target subunit 2\nDE\n")
    spec = load_target_spec("T", fasta)

    pdb = tmp_path / "raw.pdb"
    pdb.write_text(
        residue(1, "ASP", "X", 1)
        + residue(5, "GLU", "X", 2)
        + "TER\n"
        + residue(9, "ALA", "Y", 1)
        + residue(13, "CYS", "Y", 2)
        + "END\n"
    )

    out_pdb = tmp_path / "clean.pdb"
    record = clean_model(RawModel("m1", str(pdb), "pdb"), spec, out_pdb)
    text = out_pdb.read_text()

    assert record["cleaned_residue_count"] == 4
    assert " A   1" in text
    assert " A   2" in text
    assert " B   1" in text
    assert " B   2" in text
    assert text.index("ALA A   1") < text.index("ASP B   1")


def test_clean_model_canonicalizes_six_chain_homomer_copy_permutation(tmp_path: Path):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(
        ">Ag1\nAC\n>Ag2\nAC\n>H1\nDE\n>H2\nDE\n>L1\nFG\n>L2\nFG\n"
    )
    spec = load_target_spec("T355", fasta)
    raw = tmp_path / "raw.pdb"
    sequences = {
        "Q": (("PHE", "GLY"), 1),
        "X": (("ALA", "CYS"), 10),
        "R": (("ASP", "GLU"), 20),
        "Y": (("ALA", "CYS"), 30),
        "S": (("PHE", "GLY"), 40),
        "Z": (("ASP", "GLU"), 50),
    }
    serial = 1
    text = ""
    for chain, (resnames, start) in sequences.items():
        for offset, resname in enumerate(resnames):
            text += residue(serial, resname, chain, start + offset)
            serial += 4
        text += "TER\n"
    raw.write_text(text + "END\n")

    output = tmp_path / "clean.pdb"
    record = clean_model(
        RawModel("six", str(raw), "pdb"), spec, output,
        min_identity=0.99, min_raw_coverage=0.95, min_target_coverage=0.95,
    )

    assert [entry["target_chain_id"] for entry in record["chain_mapping"]] == list("ABCDEF")
    assert [entry["raw_chain_id"] for entry in record["chain_mapping"]] == ["X", "Y", "R", "Z", "Q", "S"]
    atom_chains = []
    for line in output.read_text().splitlines():
        if line.startswith("ATOM") and line[21] not in atom_chains:
            atom_chains.append(line[21])
    assert atom_chains == list("ABCDEF")


def test_clean_model_reports_target_coordinate_offsets_for_truncated_chain(tmp_path: Path):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">target subunit 1\nGGACDE\n")
    spec = load_target_spec("T", fasta)

    pdb = tmp_path / "raw.pdb"
    pdb.write_text(
        residue(1, "ALA", "X", 1)
        + residue(5, "CYS", "X", 2)
        + residue(9, "ASP", "X", 3)
        + residue(13, "GLU", "X", 4)
        + "END\n"
    )

    out_pdb = tmp_path / "clean.pdb"
    record = clean_model(RawModel("m1", str(pdb), "pdb"), spec, out_pdb, min_target_coverage=0.5)
    text = out_pdb.read_text()

    assert "ALA A   3" in text
    assert record["chain_alignment_methods"] == "A:substring"
    assert record["chain_target_starts"] == "A:3"
    assert record["chain_target_ends"] == "A:6"
    assert record["chain_n_terminal_missing"] == "A:2"
    assert record["chain_c_terminal_missing"] == "A:0"


def test_clean_model_rejects_unrepresented_homomer_copy(tmp_path: Path):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">target subunit 1\nAC\n")
    spec = load_target_spec("T", fasta)

    pdb = tmp_path / "raw.pdb"
    pdb.write_text(
        residue(1, "ALA", "X", 1)
        + residue(5, "CYS", "X", 2)
        + "TER\n"
        + residue(9, "ALA", "Y", 1)
        + residue(13, "CYS", "Y", 2)
        + "END\n"
    )

    with pytest.raises(CleaningError, match="FASTA may omit homomer copies"):
        clean_model(RawModel("m1", str(pdb), "pdb"), spec, tmp_path / "clean.pdb")


def test_clean_models_streams_multi_model_pdb_and_preserves_md5(tmp_path: Path):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">target subunit 1\nAC\n")
    spec = load_target_spec("T", fasta)

    pdb = tmp_path / "Target.pdb"
    pdb.write_text(
        "REMARK   9 MODEL        0 MD5 913ac7b5edda080f3758373ae8a8fc09\n"
        "REMARK   9 MODEL        1 MD5 e05ee793be8c108df67f7858ec5de64f\n"
        "MODEL        0\n"
        + residue(1, "ALA", "X", 1)
        + residue(5, "CYS", "X", 2)
        + "ENDMDL\n"
        "MODEL        1\n"
        + residue(9, "ALA", "Y", 10)
        + residue(13, "CYS", "Y", 11)
        + "ENDMDL\n"
    )

    out_dir = tmp_path / "out"
    records = clean_models(spec, pdb, out_dir)
    manifest = read_jsonl(out_dir / "manifests" / "cleaning.jsonl")

    assert len(records) == 2
    assert len(manifest) == 2
    assert {record["capri_md5"] for record in manifest} == {
        "913ac7b5edda080f3758373ae8a8fc09",
        "e05ee793be8c108df67f7858ec5de64f",
    }
    assert (out_dir / "cleaned" / "af2rank_input" / "T" / "Target_model_0000.pdb").exists()
    assert (out_dir / "cleaned" / "af2rank_input" / "T" / "Target_model_0001.pdb").exists()


def test_clean_batch_and_merge_clean_manifest(tmp_path: Path):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">target subunit 1\nAC\n")
    spec = load_target_spec("T", fasta)

    pdb = tmp_path / "raw.pdb"
    pdb.write_text(residue(1, "ALA", "X", 1) + residue(5, "CYS", "X", 2) + "END\n")
    raw_model = RawModel("m1", str(pdb), "pdb")
    batch_root = tmp_path / "batches"
    batch_root.mkdir()
    batch_manifest = write_raw_batch_manifest(batch_root, 0, [raw_model])

    out_dir = tmp_path / "out"
    spec.write_json(out_dir / "target_spec.json")
    records = clean_batch(spec, batch_manifest, out_dir, run_label="target_batch_000")
    merged = merge_clean_batch_manifests(out_dir, target_prefix="target")

    assert len(records) == 1
    assert len(merged) == 1
    assert (out_dir / "manifests" / "cleaning_target_batch_000.jsonl").exists()
    assert read_jsonl(out_dir / "manifests" / "cleaning.jsonl")[0]["model_id"] == "m1"
    assert (out_dir / "cleaned" / "af2rank_decoy_list.txt").read_text() == "T m1.pdb\n"


def test_clean_batch_streams_compressed_multi_model_pdb(tmp_path: Path):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">target subunit 1\nAC\n")
    spec = load_target_spec("T", fasta)

    pdb = tmp_path / "Target.pdb.gz"
    with gzip.open(pdb, "wt") as handle:
        handle.write(
            "REMARK   9 MODEL        0 MD5 913ac7b5edda080f3758373ae8a8fc09\n"
            "REMARK   9 MODEL        1 MD5 e05ee793be8c108df67f7858ec5de64f\n"
            "MODEL        0\n"
            + residue(1, "ALA", "X", 1)
            + residue(5, "CYS", "X", 2)
            + "ENDMDL\n"
            "MODEL        1\n"
            + residue(9, "ALA", "Y", 10)
            + residue(13, "CYS", "Y", 11)
            + "ENDMDL\n"
        )

    batch_root = tmp_path / "batches"
    batch_root.mkdir()
    batch_manifest = write_raw_batch_manifest(batch_root, 0, discover_models(pdb))

    with patch("af2rank_pipeline.cleaning.load_structure", side_effect=AssertionError("unexpected rescan")):
        records = clean_batch(spec, batch_manifest, tmp_path / "out", run_label="target_batch_000")

    assert [record["model_number"] for record in records] == [0, 1]
    assert {record["capri_md5"] for record in records} == {
        "913ac7b5edda080f3758373ae8a8fc09",
        "e05ee793be8c108df67f7858ec5de64f",
    }
