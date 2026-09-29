# Third-party components

## Included in this repository

- The `DockQ/` directory contains modified [DockQ](https://github.com/wallnerlab/DockQ) source from tag `v2.1.3` (commit `d9cbb1940bb0f42db3257f7da3b0e96f162b94d9`). Its [MIT license](DockQ/src/DockQ/LICENSE) is included with the packaged code. The changes expose contact and interface residue sets in JSON, add fallbacks for optional dependencies, and adjust parsing and file-handle behavior. See the [DockQ v2 paper](https://doi.org/10.1093/bioinformatics/btae586).
- The `examples/1brs` structure and sequences are derived from [PDB 1BRS](https://doi.org/10.2210/pdb1BRS/pdb). PDB archive data are available under [CC0](https://www.rcsb.org/pages/usage-policy).

## Methods and external components

- The scoring approach follows [AF2Rank](https://github.com/jproney/AF2Rank) by Roney and Ovchinnikov ([paper](https://doi.org/10.1103/PhysRevLett.129.238101)). The upstream AF2Rank repository has its [own MIT license](https://github.com/jproney/AF2Rank/blob/master/LICENSE). AF2Rank is not installed as a separate dependency or bundled as a source tree here.
- Inference uses [ColabDesign](https://github.com/sokrypton/ColabDesign), installed from the commit pinned in `pyproject.toml`. That revision includes a [Beer-Ware notice at `af/LICENSE.txt`](https://github.com/sokrypton/ColabDesign/blob/094e2cb3603dee7d99846e0977736bd943c830c2/af/LICENSE.txt). ColabDesign source is not included in this repository.
- [AlphaFold 2 code](https://github.com/google-deepmind/alphafold/blob/main/LICENSE) is Apache 2.0, while its [model parameters](https://github.com/google-deepmind/alphafold/blob/main/README.md#model-parameters) are CC BY 4.0. The setup helper downloads parameters; they are not included in this repository.
- The pipeline calculates [actifpTM](https://doi.org/10.1093/bioinformatics/btaf107) metrics. [TMscore](https://zhanggroup.org/TM-score/) is an optional external executable and is not distributed here.

The project-owned code is [MIT licensed](LICENSE). The bundled DockQ code retains its separate [MIT notice](DockQ/src/DockQ/LICENSE); external dependencies and model parameters remain under their upstream terms.
