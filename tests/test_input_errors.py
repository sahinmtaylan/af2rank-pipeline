import ast
import contextlib
import csv
import gzip
import html
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from af2rank_pipeline.cli import main
from af2rank_pipeline.exceptions import CleaningError
from af2rank_pipeline.ingest import discover_models

ATOM = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("text, reason", [
    (ATOM + "ENDMDL\n" + ATOM + "ENDMDL\n", "ENDMDL at line 2 has no preceding MODEL header"),
    ("MODEL        1\nMODEL        2\nENDMDL\n", "MODEL at line 2 appears before ENDMDL"),
    ("MODEL        1\n" + ATOM, "MODEL at line 1 has no ENDMDL before the end of the file"),
    ("MODEL        1\n" + ATOM + "END\n", "no ENDMDL before END at line 3"),
    (ATOM + "MODEL        1\nENDMDL\n", "Atoms at line 1 appear before the first MODEL header"),
    ("MODEL        1\nENDMDL\n" + ATOM, "Atoms at line 3 appear outside a MODEL/ENDMDL block"),
    ("MODEL        X\nENDMDL\n", "MODEL at line 1 has no valid integer model number"),
])
def test_rejects_malformed_ensemble_with_actionable_error(tmp_path, compressed, text, reason):
    path = tmp_path / ("ensemble.pdb.gz" if compressed else "ensemble.pdb")
    if compressed:
        with gzip.open(path, "wt") as handle:
            handle.write(text)
    else:
        path.write_text(text)
    with pytest.raises(CleaningError) as error:
        discover_models(path)
    message = str(error.value)
    assert path.name in message
    assert reason in message
    assert "start with a numbered MODEL record and end with ENDMDL" in message
    assert "separate PDB files" in message


def test_valid_single_and_ensemble_inputs_remain_supported(tmp_path):
    path = tmp_path / "models.pdb"
    path.write_text(ATOM + "END\n")
    assert len(discover_models(path)) == 1
    path.write_text("MODEL        7\n" + ATOM + "ENDMDL\nEND\n")
    assert len(discover_models(path)) == 1
    path.write_text("MODEL        0\n" + ATOM + "ENDMDL\nMODEL        8\n" + ATOM + "ENDMDL\nEND\n")
    assert [m.model_number for m in discover_models(path)] == [0, 8]


def test_run_target_rejects_invalid_ensemble_before_inference(tmp_path, monkeypatch, capsys):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">A\nA\n")
    pdb = tmp_path / "ensemble.pdb"
    pdb.write_text(ATOM + "ENDMDL\n" + ATOM + "ENDMDL\n")
    inference = Mock()
    monkeypatch.setattr("af2rank_pipeline.af2rank_runner.run_af2rank_stage", inference)
    assert main([
        "run-target", "--target", "T", "--fasta", str(fasta),
        "--models", str(pdb), "--out", str(tmp_path / "out"),
        "--params", str(tmp_path / "absent-weights"),
    ]) == 1
    inference.assert_not_called()
    error = capsys.readouterr().err
    assert error.startswith("af2rank-pipeline: error: ensemble.pdb: ENDMDL at line 2")
    assert "Traceback" not in error


def notebook_runner(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    notebook = json.loads((repo / "notebooks/colab_demo.ipynb").read_text())
    source = "".join(next(c for c in notebook["cells"] if c["metadata"].get("id") == "scoring-helpers")["source"])
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "run_with_progress")
    displayed = []
    widgets = SimpleNamespace(
        IntProgress=lambda **kw: SimpleNamespace(**kw),
        HTML=lambda value: SimpleNamespace(value=value),
        Layout=lambda **kw: kw,
        Output=contextlib.nullcontext,
        Accordion=lambda **kw: SimpleNamespace(**kw, set_title=lambda *args: None),
        VBox=lambda children: children,
    )
    namespace = dict(repo=repo, pipeline_env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                     widgets=widgets, display=displayed.append, subprocess=subprocess,
                     re=re, csv=csv, html=html, os=os, signal=signal)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "colab_demo.ipynb", "exec"), namespace)
    return namespace["run_with_progress"], displayed


def test_notebook_surfaces_cli_error_and_keeps_full_log(tmp_path):
    run, displayed = notebook_runner(tmp_path)
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">A\nA\n")
    pdb = tmp_path / "ensemble.pdb"
    pdb.write_text(ATOM + "ENDMDL\n")
    out = tmp_path / "out"
    command = [sys.executable, "-m", "af2rank_pipeline.cli", "clean", "--target", "T",
               "--fasta", str(fasta), "--models", str(pdb), "--out", str(out)]
    with pytest.raises(RuntimeError, match="ENDMDL at line 2 has no preceding MODEL header"):
        run(command, "T", 1, out)
    status, progress, panel = displayed[0]
    assert "ENDMDL at line 2 has no preceding MODEL header" in status.value
    assert "separate PDB files" in status.value
    assert progress.bar_style == "danger"
    assert panel.selected_index == 0
    assert "af2rank-pipeline: error:" in (out / "logs/notebook.log").read_text()


def test_notebook_unexpected_subprocess_failure_retains_diagnostics(tmp_path):
    run, displayed = notebook_runner(tmp_path)
    out = tmp_path / "out"
    command = [sys.executable, "-c", "import sys; print('unexpected <failure>'); sys.exit(7)"]
    with pytest.raises(RuntimeError, match="Pipeline exited with status 7"):
        run(command, "T", 1, out)
    assert "Pipeline exited with status 7" in displayed[0][0].value
    assert "unexpected &lt;failure&gt;" in displayed[0][0].value
    assert "unexpected <failure>" in (out / "logs/notebook.log").read_text()
