from types import SimpleNamespace

import numpy as np
import pytest

from af2rank_pipeline.actifptm_runner import (
    build_asym_id_from_lengths,
    chain_ranges_from_asym_id,
    compute_pair_metrics,
    get_asym_id,
)


def runner_with_inputs(inputs):
    return SimpleNamespace(model=SimpleNamespace(_inputs=inputs))


@pytest.mark.parametrize("batched", [False, True])
def test_chain_ids_follow_prepared_residues_instead_of_pdb_counts(batched):
    expected = build_asym_id_from_lengths([54, 55])
    inputs = {
        "asym_id": expected[None, :] if batched else expected,
        "batch": {"aatype": np.zeros(109), "asym_id": np.zeros(110)},
    }
    # A nonexistent path proves the raw structure is not reread for chain IDs.
    actual = get_asym_id(runner_with_inputs(inputs), "unused.pdb", "A,B")
    np.testing.assert_array_equal(actual, expected)
    assert chain_ranges_from_asym_id(actual, ["A", "B"]) == {"A": (0, 53), "B": (54, 108)}


def test_legacy_batch_chain_ids_remain_supported():
    expected = np.array([0, 1, 1])
    actual = get_asym_id(runner_with_inputs({"batch": {"asym_id": expected}}), "unused.pdb", "A,B")
    np.testing.assert_array_equal(actual, expected)


def test_pair_metrics_accept_prepared_109_residue_features():
    pytest.importorskip("jax")
    pytest.importorskip("scipy")
    asym_id = build_asym_id_from_lengths([54, 55])
    actual = get_asym_id(runner_with_inputs({"asym_id": asym_id}), "unused.pdb", "A,B")
    outputs = {
        "distogram": {"logits": np.zeros((109, 109, 3)), "bin_edges": np.linspace(0, 31, 3)},
        "predicted_aligned_error": {"logits": np.zeros((109, 109, 3)), "breaks": np.linspace(0, 31, 4)},
    }
    metrics = compute_pair_metrics(outputs, actual, ["A", "B"])
    assert set(metrics["per_chain_ptm"]) == {"A", "B"}
    assert set(metrics["pairwise_actifptm"]) == {"A_B"}
    assert np.isfinite(metrics["actifptm"])


def test_colabdesign_preparation_with_missing_backbone_nitrogen(tmp_path):
    prep = pytest.importorskip("colabdesign.af.prep")
    lines = []
    for chain in ["A", "B"]:
        for residue in [1, 2]:
            for atom in ["N", "CA"]:
                if chain == "A" and residue == 1 and atom == "N":
                    continue
                lines.append(
                    f"ATOM  {len(lines)+1:5d} {atom:>4} ALA {chain}{residue:4d}    "
                    f"{float(residue):8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{0.:6.2f}           {atom[0]:>2}\n"
                )
    pdb = tmp_path / "missing_n.pdb"
    pdb.write_text("".join(lines) + "END\n")
    prepared = prep.prep_pdb(str(pdb), chain="A,B", ignore_missing=True)
    assert prepared["lengths"] == [1, 2]
    inputs = {"batch": prepared["batch"], **prep.get_multi_id(prepared["lengths"])}
    actual = get_asym_id(runner_with_inputs(inputs), str(pdb), "A,B")
    np.testing.assert_array_equal(actual, [0, 1, 1])
    assert len(actual) == len(prepared["batch"]["aatype"])
