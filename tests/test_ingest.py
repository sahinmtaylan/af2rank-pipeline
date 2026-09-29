import gzip
from pathlib import Path

from af2rank_pipeline.ingest import discover_models


def test_discovers_multi_model_pdb(tmp_path: Path):
    pdb = tmp_path / "Target.pdb"
    pdb.write_text(
        "REMARK   9 MODEL        0 MD5 913ac7b5edda080f3758373ae8a8fc09\n"
        "MODEL        0\nENDMDL\n"
        "MODEL        1\nENDMDL\n"
    )

    models = discover_models(pdb)

    assert [model.model_number for model in models] == [0, 1]
    assert models[0].capri_md5 == "913ac7b5edda080f3758373ae8a8fc09"


def test_discovers_compressed_multi_model_pdb(tmp_path: Path):
    pdb = tmp_path / "Target.pdb.gz"
    with gzip.open(pdb, "wt") as handle:
        handle.write(
            "REMARK   9 MODEL        0 MD5 913ac7b5edda080f3758373ae8a8fc09\n"
            "MODEL        0\nENDMDL\n"
            "MODEL        1\nENDMDL\n"
        )

    models = discover_models(pdb)

    assert [model.model_id for model in models] == ["Target_model_0000", "Target_model_0001"]
    assert [model.model_number for model in models] == [0, 1]
