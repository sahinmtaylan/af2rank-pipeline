# AF2Rank Pipeline

AF2Rank Pipeline scores existing protein-complex structures. It standardizes chains and residues against a target FASTA, runs AF2Rank through ColabDesign, calculates actifpTM and interface metrics, and combines the results in a CSV for model comparison.

## Try it in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahinmtaylan/af2rank-pipeline/blob/main/notebooks/colab_demo.ipynb)

Select a GPU runtime to run the included 1BRS example or upload your own FASTA and structures.

## Run locally

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from a clone of this repository. The project uses Python 3.10 and a committed lockfile:

```bash
uv python install 3.10
uv sync --locked --all-extras
source .venv/bin/activate
```

The lockfile installs CPU JAX. A GPU is **not required** for the bundled single-model example: the full 199-residue 1BRS run has been tested on CPU. Larger structures or many models may take much longer; use a GPU for those workloads.

Full scoring needs AlphaFold model parameters, but not AlphaFold's sequence databases. Download the default model-2 pTM weight (373 MB) from the [official parameter archive](https://github.com/google-deepmind/alphafold#model-parameters):

```bash
python env/download_default_params.py params
```

The helper downloads just the required file and verifies its checksum. `params/` is excluded from version control. If you already have a parameter directory, use it instead. A different model number or `--model-mode alphafold-multimer` needs its corresponding weight; the [official download script](https://github.com/google-deepmind/alphafold/blob/main/scripts/download_alphafold_params.sh) provides the complete parameter bundle.

Run the included 1BRS complex:

```bash
af2rank-pipeline run-target \
  --target 1brs \
  --fasta examples/1brs/target.fasta \
  --models examples/1brs/model.pdb \
  --out runs/1brs \
  --params params
```

The result table is `runs/1brs/manifests/final_models.csv`; the scored structure and PAE data are under `runs/1brs/af2rank/`. The [example notes](examples/1brs/README.md) describe the source of the bundled data. Use a new output directory for each independent run.

### NVIDIA GPU on Linux

For an NVIDIA GPU with a compatible CUDA 12 driver, install the matching JAX wheel **after** `uv sync`:

```bash
uv pip install --python .venv/bin/python \
  --constraint env/cuda12-constraints.txt \
  --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  'jax[cuda12_pip]==0.4.20'
python -c "import jax; print(jax.devices())"
```

Check that JAX reports a GPU, then run `af2rank-pipeline` from the activated environment as above. Running `uv sync` or `uv run` afterward can restore the locked CPU wheel. This CUDA overlay has completed the bundled example in hosted Colab; GPU drivers and libraries vary across machines. Follow [JAX's accelerator guidance](https://docs.jax.dev/en/latest/installation.html) if device detection fails. On a shared cluster, use the site's approved environment and run inference inside a scheduled GPU allocation.

## Use your own models

Provide one FASTA record per physical protein chain. Homomer copies need separate records with unique headers; record order determines the canonical chain order. Use standard amino-acid letters.

```fasta
>chain_A
ACDEFGHIKLMNPQRSTVWY
>chain_B
ACDEFGHIKLMNPQRSTVWY
```

`--models` accepts a PDB or mmCIF file, a gzipped PDB, a multi-model PDB, or a directory of structures. Nonprotein components are not retained during cleaning. To score your own models:

```bash
af2rank-pipeline run-target \
  --target my_complex \
  --fasta /path/to/target.fasta \
  --models /path/to/models \
  --out runs/my_complex \
  --params params
```

The default run uses pTM model 2, one recycle, and one inference iteration. It masks template sequence and side chains. Use `af2rank-pipeline run-target --help` for other model and scoring settings. `--tm-exec` optionally points to a working [TMscore](https://zhanggroup.org/TM-score/) executable; without it, TM-score fields and the composite score are empty.

## Results

| Path in the output directory | Contents |
| --- | --- |
| `target_spec.json` | Target chains and sequences |
| `cleaned/af2rank_input/` | Standardized input structures |
| `manifests/cleaning.jsonl` | Chain alignment, coverage, and source provenance |
| `af2rank/` | AF2Rank scores, scored PDBs, and PAE data |
| `additional_metrics/` | Input–output DockQ and interface PAE metrics |
| `manifests/final_models.csv` | Combined scores and provenance for each model |
| `logs/failures.jsonl` | Stage errors, if any |

DockQ uses the cleaned input as its reference, so a high value means the interface remained similar after AF2Rank inference. It does not establish docking accuracy. Check the output row count and any failure log before comparing models.

## Other workflows

`af2rank-pipeline clean` runs the input-cleaning stage without model parameters or inference. The CLI also provides staged commands for batching, AF2Rank inference, DockQ, aggregation, and cleanup; use `af2rank-pipeline --help` and each subcommand's `--help`. `--resume` can reuse completed cleaning and AF2Rank results when the model set and scoring configuration match; start a new output directory if inputs or weights change, because resume does not hash their contents.

To check a local installation or contribute changes:

```bash
uv run --locked --all-extras python -m pytest -q
```

## License

The project-owned code is [MIT licensed](LICENSE). See [third-party components and citations](THIRD_PARTY.md) for bundled code, example data, and external dependencies.
