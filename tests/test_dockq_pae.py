import csv
import gzip
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from af2rank_pipeline import dockq_pae
from af2rank_pipeline.aggregate import summarize_pae_metrics


def _spec(tmp_path: Path) -> dockq_pae.RunSpec:
    native = tmp_path / "native.pdb"
    native.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  CA  GLY B   1       1.000   0.000   0.000  1.00 20.00           C\n"
        "END\n"
    )
    model = tmp_path / "model.pdb"
    model.write_text(native.read_text())
    pae = tmp_path / "pae.json"
    pae.write_text(json.dumps({"pae": [[0, 2], [3, 0]]}))
    return dockq_pae.RunSpec("m1", "batch_000", "m1.pdb", native, model, pae)


def test_process_specs_writes_native_contact_metrics(tmp_path: Path, monkeypatch):
    spec = _spec(tmp_path)

    def fake_run_dockq(_spec, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "best_dockq": 0.4,
            "GlobalDockQ": 0.4,
            "best_result": {
                "AB": {
                    "chain1": "A",
                    "chain2": "B",
                    "DockQ": 0.4,
                    "native_contacts": {
                        "chain1": [{"resnum": 1, "resname": "ALA"}],
                        "chain2": [{"resnum": 1, "resname": "GLY"}],
                    },
                },
            },
        }))
        return path

    monkeypatch.setattr(dockq_pae, "run_dockq", fake_run_dockq)
    out = tmp_path / "metrics"
    assert dockq_pae.process_specs([spec], out) == 0
    with gzip.open(out / "pae_raw.csv.gz", "rt", newline="") as handle:
        pae_rows = list(csv.DictReader(handle))
    assert [row["pae"] for row in pae_rows] == ["2.000", "3.000"]
    assert [row["direction"] for row in pae_rows] == ["A-B", "B-A"]
    with (out / "dockq_metrics.csv").open(newline="") as handle:
        dockq_rows = list(csv.DictReader(handle))
    assert dockq_rows[0]["DockQ"] == "0.4"
    assert not list((out / ".tmp").glob("*.pae_rows.csv.gz"))


def test_process_specs_rejects_missing_inputs(tmp_path: Path):
    spec = _spec(tmp_path)
    spec.pae.unlink()
    with pytest.raises(ValueError, match="missing pae"):
        dockq_pae.process_specs([spec], tmp_path / "metrics")


def _backbone_spec(tmp_path, omitted):
    spec = replace(_spec(tmp_path), pae_chain_order="A,B")
    # Native file order differs from AF2Rank's requested chain order, and
    # native numbering has gaps. Both must survive the PAE mapping.
    residues = [("B", 5), ("B", 6), ("A", 10), ("A", 20), ("A", 30)]
    lines = []
    for chain, resnum in residues:
        for atom in ("N", "CA", "C", "O"):
            if (chain, resnum) in omitted and atom == "N":
                continue
            lines.append(
                f"ATOM  {len(lines)+1:5d} {atom:>4} ALA {chain}{resnum:4d}    "
                f"{float(resnum):8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{0.:6.2f}           {atom[0]:>2}\n"
            )
    spec.native.write_text("".join(lines) + "END\n")
    spec.model.write_text(spec.native.read_text())
    data = {"best_result": {"AB": {
        "chain1": "A", "chain2": "B",
        "native_contacts": {
            "chain1": [{"resnum": n, "resname": "ALA"} for n in (10, 20, 30)],
            "chain2": [{"resnum": n, "resname": "ALA"} for n in (5, 6)],
        },
    }}}
    return spec, data


@pytest.mark.parametrize(
    "omitted", [set(), {("A", 10)}, {("A", 20)}, {("B", 5)}, {("A", 20), ("B", 5)}],
)
def test_pae_rows_preserve_native_ids_after_backbone_omissions(tmp_path, capsys, omitted):
    spec, data = _backbone_spec(tmp_path, omitted)
    retained = [
        key for key in [("A", 10), ("A", 20), ("A", 30), ("B", 5), ("B", 6)]
        if key not in omitted
    ]
    indices = {key: i for i, key in enumerate(retained)}
    matrix = np.arange(len(retained)**2).reshape(len(retained), len(retained))
    rows = list(dockq_pae.iter_pae_rows(spec, data, matrix))
    actual = {
        (row["scored_chain"], row["scored_resnum"], row["aligned_chain"], row["aligned_resnum"]): row["pae"]
        for row in rows
    }
    expected = {
        (*scored, *aligned): matrix[indices[scored], indices[aligned]]
        for scored in retained for aligned in retained if scored[0] != aligned[0]
    }
    assert actual == expected
    assert len(rows) == len(expected)
    warning = capsys.readouterr().err
    if omitted:
        assert f"ColabDesign omitted {len(omitted)} residue(s)" in warning
        assert all(f"{chain}:{resnum}" in warning for chain, resnum in omitted)
    else:
        assert warning == ""

    # Aggregation must use exactly these retained contact pairs, with no
    # fabricated PAE values or changes to the existing result columns.
    metrics = tmp_path / "additional_metrics"
    metrics.mkdir()
    with gzip.open(metrics / "pae_raw.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dockq_pae.PAE_COMPACT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_pae_metrics(tmp_path)["m1"]
    assert summary["pae_contacts_count"] == len(expected)
    assert summary["pae_contacts_mean"] == f"{np.mean(list(expected.values())):.4f}"


def test_filtered_mapping_matches_colabdesign_preparation(tmp_path):
    prep = pytest.importorskip("colabdesign.af.prep")
    spec, _ = _backbone_spec(tmp_path, {("A", 20), ("B", 5)})
    prepared = prep.prep_pdb(str(spec.native), chain="A,B", ignore_missing=True)
    expected = {
        (chain, int(resnum)): i
        for i, (chain, resnum) in enumerate(zip(prepared["idx"]["chain"], prepared["idx"]["residue"]))
    }
    assert dockq_pae.build_residue_to_pae_index(spec.native, "A,B", required_atom="N") == expected


@pytest.mark.parametrize("size", [3, 6])
def test_unexplained_pae_size_mismatch_still_fails(tmp_path, capsys, size):
    spec, data = _backbone_spec(tmp_path, {("A", 20)})
    # Five input residues, four retained: no other size is explainable.
    with pytest.raises(ValueError, match="PAE size .* backbone N atoms"):
        list(dockq_pae.iter_pae_rows(spec, data, np.zeros((size, size))))
    assert capsys.readouterr().err == ""


def test_full_size_pae_retains_incomplete_residue_without_warning(tmp_path, capsys):
    spec, data = _backbone_spec(tmp_path, {("A", 20)})
    # An engine that retains incomplete residues needs no filtering.
    rows = list(dockq_pae.iter_pae_rows(spec, data, np.zeros((5, 5))))
    assert len(rows) == 12
    assert any(row["scored_chain"] == "A" and row["scored_resnum"] == 20 for row in rows)
    assert capsys.readouterr().err == ""
