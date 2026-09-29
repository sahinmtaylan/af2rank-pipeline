from pathlib import Path

import pytest

from af2rank_pipeline.af2rank_runner import AF2RankConfig, merge_af2rank_batch_manifests, run_af2rank_stage
from af2rank_pipeline.manifests import append_jsonl, read_csv_rows, write_csv
from af2rank_pipeline.target import load_target_spec


def test_af2rank_stage_bridges_actifptm_outputs(tmp_path: Path):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">target subunit 1\nAC\n")
    spec = load_target_spec("T", fasta)

    out = tmp_path / "run"
    cleaned = out / "cleaned" / "af2rank_input" / "T"
    cleaned.mkdir(parents=True)
    input_pdb = cleaned / "model_0001.pdb"
    input_pdb.write_text("END\n")
    append_jsonl(
        out / "manifests" / "cleaning.jsonl",
        {"stage": "clean", "status": "ok", "model_id": "model_0001", "cleaned_path": str(input_pdb)},
    )

    fake_runner = tmp_path / "fake_actifptm.py"
    fake_runner.write_text(
        "import argparse, csv, gzip, json, numpy as np\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--output_dir', required=True); "
        "p.add_argument('--target', required=True); p.add_argument('--decoy_dir', required=True); "
        "p.add_argument('--pae_json_dir', required=True)\n"
        "p.add_argument('--chain'); p.add_argument('--model_mode'); "
        "p.add_argument('--model_num'); p.add_argument('--recycle'); p.add_argument('--iterations')\n"
        "p.add_argument('--params'); p.add_argument('--colabdesign'); p.add_argument('--tm')\n"
        "p.add_argument('--mask_sequence', action='store_true'); "
        "p.add_argument('--mask_sidechains', action='store_true'); "
        "p.add_argument('--mask_interchain', action='store_true'); "
        "p.add_argument('--keep_raw', action='store_true'); p.add_argument('--no_pae_json_gzip', action='store_true')\n"
        "a=p.parse_args(); out=Path(a.output_dir); (out/'scored_pdbs').mkdir(parents=True); "
        "pae=Path(a.pae_json_dir); pae.mkdir(parents=True, exist_ok=True)\n"
        "(out/'scored_pdbs'/'model_0001_scored.pdb').write_text('END\\n')\n"
        "with gzip.open(pae/'model_0001.json.gz','wt') as h: json.dump({'model_id':'model_0001','pae':np.zeros((2,2)).tolist()}, h)\n"
        "with (out/f'{a.target}_af2rank_actifptm.csv').open('w', newline='') as f:\n"
        " w=csv.DictWriter(f, fieldnames=['id','target','plddt','ptm','tm_io']); "
        "w.writeheader(); w.writerow({'id':'model_0001.pdb','target':a.target,'plddt':'90','ptm':'0.5','tm_io':'0.8'})\n"
    )

    records = run_af2rank_stage(
        spec,
        out,
        AF2RankConfig(pipeline_script=str(fake_runner), python_executable="python3", tm_exec=None),
    )

    assert len(records) == 1
    assert (out / "af2rank" / "pae" / "model_0001.json.gz").exists()
    af2rank_row = read_csv_rows(out / "manifests" / "af2rank.csv")[0]
    assert af2rank_row["model_id"] == "model_0001"
    assert af2rank_row["raw_npz"] == ""
    dockq_manifest = read_csv_rows(out / "manifests" / "dockq_pae_manifest.csv")
    assert dockq_manifest[0]["native"] == str(input_pdb)
    assert dockq_manifest[0]["pae_chain_order"] == "A"

    with pytest.raises(ValueError, match="iterations must be at least 1"):
        run_af2rank_stage(
            spec,
            out,
            AF2RankConfig(pipeline_script=str(fake_runner), python_executable="python3", iterations=0),
        )

    resumed = run_af2rank_stage(
        spec,
        out,
        AF2RankConfig(pipeline_script=str(fake_runner), python_executable="python3", tm_exec=None),
        resume=True,
    )
    assert len(resumed) == 1

    with pytest.raises(ValueError, match="scoring settings changed \\(iterations\\)"):
        run_af2rank_stage(
            spec,
            out,
            AF2RankConfig(
                pipeline_script=str(fake_runner),
                python_executable="python3",
                tm_exec=None,
                iterations=2,
            ),
            resume=True,
        )

    second_pdb = cleaned / "model_0002.pdb"
    second_pdb.write_text("END\n")
    append_jsonl(
        out / "manifests" / "cleaning.jsonl",
        {"stage": "clean", "status": "ok", "model_id": "model_0002", "cleaned_path": str(second_pdb)},
    )
    with pytest.raises(ValueError, match="do not match cleaned inputs"):
        run_af2rank_stage(
            spec,
            out,
            AF2RankConfig(pipeline_script=str(fake_runner), python_executable="python3", tm_exec=None),
            resume=True,
        )


def test_merge_af2rank_batch_manifests(tmp_path: Path):
    out = tmp_path / "run"
    manifests = out / "manifests"
    write_csv(
        manifests / "af2rank_target_batch_000.csv",
        [{"model_id": "m1", "status": "ok", "plddt": "90", "actifptm_A_B": "0.7"}],
        ["model_id", "status", "plddt", "actifptm_A_B"],
    )
    write_csv(
        manifests / "dockq_pae_manifest_target_batch_000.csv",
        [{"model_id": "m1.pdb", "native": "m1.pdb", "model": "m1_scored.pdb", "pae": "m1.json"}],
        ["model_id", "native", "model", "pae"],
    )

    af2rank_rows, dockq_rows = merge_af2rank_batch_manifests(out)

    assert len(af2rank_rows) == 1
    assert len(dockq_rows) == 1
    assert read_csv_rows(manifests / "af2rank.csv")[0]["actifptm_A_B"] == "0.7"
    assert read_csv_rows(manifests / "dockq_pae_manifest.csv")[0]["model_id"] == "m1.pdb"
