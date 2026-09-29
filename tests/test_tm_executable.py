from pathlib import Path

import numpy as np
import pytest

from af2rank_pipeline.actifptm_runner import tmscore, validate_tm_executable


def test_explicit_tm_executable_must_work(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="TMscore executable not found"):
        validate_tm_executable(tmp_path / "missing")

    executable = tmp_path / "TMscore"
    executable.write_text("#!/bin/sh\nexit 2\n")
    executable.chmod(0o755)
    coordinates = np.zeros((1, 3))
    with pytest.raises(RuntimeError, match="TMscore exited with status 2"):
        tmscore(coordinates, coordinates, str(executable))

    executable.write_text("#!/bin/sh\nprintf 'TM-score = 0.5\\n'\n")
    assert tmscore(coordinates, coordinates, str(executable)) == 0.5
