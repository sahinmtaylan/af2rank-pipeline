# Modified DockQ

This directory contains [DockQ](https://github.com/wallnerlab/DockQ) source modified to expose `native_contacts`, `model_contacts`, `native_interface`, and `model_interface` in JSON output. The pipeline uses these residue sets to calculate contact and interface PAE. An upstream DockQ installation does not provide the expected JSON fields.

The included source uses a pure-Python fallback when its optional Cython extension is unavailable. Its package metadata identifies version 2.1.3. Run from the repository root with:

```bash
PYTHONPATH="$PWD/DockQ/src" python -m DockQ.DockQ --help
```

For upstream credit, the base revision, and license details, see [THIRD_PARTY.md](../THIRD_PARTY.md) and the [MIT license](src/DockQ/LICENSE).
