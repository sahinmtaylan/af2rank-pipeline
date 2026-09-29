from af2rank_pipeline.alignment import assign_chains
from af2rank_pipeline.structure import ChainRecord, ResidueRecord
from af2rank_pipeline.target import TargetChain


def chain(chain_id: str, sequence: str) -> ChainRecord:
    return ChainRecord(
        chain_id=chain_id,
        residues=[
            ResidueRecord(chain_id=chain_id, resseq=i + 1, icode="", resname="ALA", one_letter=aa)
            for i, aa in enumerate(sequence)
        ],
    )


def target(canonical_id: str, sequence: str, offset: int = 0) -> TargetChain:
    return TargetChain(
        canonical_id=canonical_id,
        header=canonical_id,
        sequence=sequence,
        length=len(sequence),
        offset=offset,
    )


def test_assigns_reordered_renamed_chains():
    raw = [chain("X", "GGG"), chain("Y", "ACD")]
    targets = [target("A", "ACD"), target("B", "GGG", offset=3)]

    alignments = assign_chains(raw, targets)

    assert [(a.raw_chain_id, a.target_chain_id) for a in alignments] == [("Y", "A"), ("X", "B")]


def test_assigns_terminally_truncated_chain():
    raw = [chain("Q", "CDE")]
    targets = [target("A", "ACDEF")]

    alignment = assign_chains(
        raw,
        targets,
        min_identity=1.0,
        min_raw_coverage=1.0,
        min_target_coverage=0.50,
    )[0]

    assert alignment.method == "substring"
    assert alignment.residue_map == {0: 1, 1: 2, 2: 3}


def test_assigns_biopython_semiglobal_alignment_with_mismatch():
    raw = [chain("Q", "ACDFF")]
    targets = [target("A", "ACDEF")]

    alignment = assign_chains(
        raw,
        targets,
        min_identity=0.80,
        min_raw_coverage=1.0,
        min_target_coverage=1.0,
    )[0]

    assert alignment.method == "biopython_semiglobal"
    assert alignment.identity == 0.80
    assert alignment.residue_map == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
