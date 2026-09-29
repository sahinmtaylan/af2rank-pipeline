from pathlib import Path

from af2rank_pipeline.target import load_target_spec


def test_load_h1311_style_fasta(tmp_path: Path):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(
        ">H1311 example, subunit 1, 3 residues;\n"
        "ACD\n"
        ">H1311 example, subunit 2, 2 residues;\n"
        "EF\n"
    )

    spec = load_target_spec("H1311", fasta)

    assert spec.target == "H1311"
    assert spec.chain_ids == "AB"
    assert spec.flat_sequence == "ACDEF"
    assert spec.chains[0].offset == 0
    assert spec.chains[1].offset == 3
    assert spec.chains[0].subunit == "1"
