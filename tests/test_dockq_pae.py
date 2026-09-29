import csv
import gzip
import json
from pathlib import Path

import pytest

from af2rank_pipeline import dockq_pae


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
