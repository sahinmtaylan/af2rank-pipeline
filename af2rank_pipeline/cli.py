from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .cleaning import clean_batch, clean_models, merge_clean_batch_manifests
from .target import load_target_spec, load_target_spec_json


def _positive_iterations(value: str) -> int:
    try:
        iterations = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("iterations must be an integer of at least 1") from exc
    if iterations < 1:
        raise argparse.ArgumentTypeError("iterations must be at least 1")
    return iterations


def _add_target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", required=True, help="Target name, e.g. example")
    parser.add_argument("--fasta", required=True, help="Per-chain target FASTA")
    parser.add_argument("--out", required=True, help="Run output directory")


def _add_clean_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", default=None, help="Directory, PDB, mmCIF, or CAPRI multi-model PDB")
    parser.add_argument("--min-identity", type=float, default=0.90)
    parser.add_argument("--min-raw-coverage", type=float, default=0.85)
    parser.add_argument("--min-target-coverage", type=float, default=0.50)


def _add_af2rank_args(parser: argparse.ArgumentParser, *, batch_options: bool = False) -> None:
    parser.add_argument("--params", required=True, help="AlphaFold parameter directory")
    parser.add_argument(
        "--colabdesign",
        default=None,
        help="Optional ColabDesign checkout; the inference extra installs a pinned version.",
    )
    if batch_options:
        parser.add_argument("--decoy-dir", default=None, help="Optional cleaned PDB directory for this AF2Rank job.")
        parser.add_argument("--run-label", default=None, help="Output label for batch jobs, e.g. target_batch_000.")
        parser.add_argument("--partial-manifest", action="store_true", help="Write batch-specific manifests for later merge.")
    parser.add_argument("--tm-exec", default=None, help="Optional TMscore executable")
    parser.add_argument("--model-mode", default="alphafold", choices=["alphafold", "alphafold-multimer"])
    parser.add_argument("--recycles", type=int, default=1)
    parser.add_argument("--model-num", type=int, default=2)
    parser.add_argument("--iterations", type=_positive_iterations, default=1)
    parser.add_argument("--no-mask-sequence", action="store_true")
    parser.add_argument("--no-mask-sidechains", action="store_true")
    parser.add_argument("--mask-interchain", action="store_true")
    parser.add_argument("--keep-raw", action="store_true", help="Save raw AF2Rank NPZ debug tensors.")


def _af2rank_config(args: argparse.Namespace) -> AF2RankConfig:
    from .af2rank_runner import AF2RankConfig

    return AF2RankConfig(
        params=args.params,
        colabdesign=args.colabdesign,
        decoy_dir=getattr(args, "decoy_dir", None),
        run_label=getattr(args, "run_label", None),
        partial_manifest=getattr(args, "partial_manifest", False),
        tm_exec=args.tm_exec,
        recycle=args.recycles,
        model_num=args.model_num,
        model_mode=args.model_mode,
        version="v3" if args.model_mode == "alphafold-multimer" else "ptm",
        iterations=args.iterations,
        mask_sequence=not args.no_mask_sequence,
        mask_sidechains=not args.no_mask_sidechains,
        mask_interchain=args.mask_interchain,
        keep_raw=args.keep_raw,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="af2rank-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_target = subparsers.add_parser("run-target", help="Run the full target pipeline")
    _add_target_args(run_target)
    _add_clean_args(run_target)
    _add_af2rank_args(run_target)
    run_target.add_argument(
        "--dockq-src",
        default=None,
        help="Path to the bundled modified DockQ src. Defaults to <repo>/DockQ/src.",
    )
    run_target.add_argument("--workers", type=int, default=1, help="Concurrent DockQ jobs")
    run_target.add_argument("--resume", action="store_true")
    run_target.add_argument("--skip-dockq", action="store_true")

    clean = subparsers.add_parser("clean", help="Clean and canonicalize raw models")
    _add_target_args(clean)
    _add_clean_args(clean)
    clean.add_argument("--batch-manifest", default=None, help="Optional raw model batch CSV to clean.")
    clean.add_argument("--run-label", default=None, help="Batch run label for batch-local manifests.")
    clean.add_argument("--resume", action="store_true")

    merge_clean = subparsers.add_parser("merge-clean", help="Merge per-batch clean manifests")
    merge_clean.add_argument("--out", required=True, help="Run output directory")
    merge_clean.add_argument("--target-prefix", default=None, help="Optional target prefix to filter batch manifests")

    af2rank = subparsers.add_parser("af2rank", help="Run AF2Rank on cleaned models")
    af2rank.add_argument("--target-spec", required=True, help="Path to target_spec.json")
    af2rank.add_argument("--out", required=True, help="Run output directory")
    _add_af2rank_args(af2rank, batch_options=True)
    af2rank.add_argument("--resume", action="store_true")

    dockq = subparsers.add_parser("dockq-io", help="Compare AF2Rank input/output PDBs with DockQ")
    dockq.add_argument("--target-spec", required=True, help="Path to target_spec.json")
    dockq.add_argument("--out", required=True, help="Run output directory")
    dockq.add_argument(
        "--dockq-src",
        default=None,
        help="Path to the bundled modified DockQ src. Defaults to <repo>/DockQ/src.",
    )
    dockq.add_argument("--jobs", type=int, default=1, help="Concurrent DockQ runs")

    aggregate = subparsers.add_parser("aggregate", help="Aggregate stage manifests into final_models.csv")
    aggregate.add_argument("--out", required=True, help="Run output directory")
    aggregate.add_argument("--allow-partial", action="store_true", help="Write exploratory tables despite missing stage results")

    merge_af2rank = subparsers.add_parser(
        "merge-af2rank", help="Merge per-batch AF2Rank manifests into root manifests"
    )
    merge_af2rank.add_argument("--out", required=True, help="Run output directory")

    make_batches_cmd = subparsers.add_parser("make-batches", help="Discover raw models and split into shared batch manifests")
    make_batches_cmd.add_argument("--target", required=True, help="Target name, e.g. example")
    make_batches_cmd.add_argument("--fasta", required=True, help="Per-chain target FASTA")
    make_batches_cmd.add_argument("--models", required=True, help="Directory, PDB, mmCIF, or CAPRI multi-model PDB")
    make_batches_cmd.add_argument("--out", required=True, help="Run output directory")
    make_batches_cmd.add_argument("--batch-root", required=True, help="Output batch root")
    make_batches_cmd.add_argument("--num-batches", type=int, required=True)
    make_batches_cmd.add_argument("--overwrite", action="store_true")

    materialize = subparsers.add_parser("materialize-af2rank-batch", help="Copy cleaned PDBs for one raw batch to a local AF2Rank directory")
    materialize.add_argument("--target-spec", required=True, help="Path to target_spec.json")
    materialize.add_argument("--out", required=True, help="Run output directory")
    materialize.add_argument("--batch-manifest", required=True, help="Raw model batch CSV")
    materialize.add_argument("--dest-dir", required=True, help="Destination directory for cleaned PDB copies")

    batch_cleanup = subparsers.add_parser("batch-cleanup", help="Clean up one completed AF2Rank batch")
    batch_cleanup.add_argument("--out", required=True, help="Run output directory")
    batch_cleanup.add_argument("--target-prefix", required=True)
    batch_cleanup.add_argument("--batch-id", required=True, help="Batch id, e.g. 000")

    cleanup = subparsers.add_parser("cleanup", help="Archive/remove heavy AF2Rank intermediates")
    cleanup.add_argument("--out", required=True, help="Run output directory")
    cleanup.add_argument("--archive", action="store_true")
    cleanup.add_argument("--no-remove", action="store_true")
    cleanup.add_argument("--remove-pae-json", action="store_true")
    cleanup.add_argument("--include-scored-pdbs", action="store_true")
    cleanup.add_argument("--no-gzip-pae-json", action="store_true")
    cleanup.add_argument("--archive-name", default="heavy_intermediates.tar.gz")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "clean":
        target_spec = load_target_spec(args.target, args.fasta)
        if args.batch_manifest:
            if not args.run_label:
                parser.error("--run-label is required with --batch-manifest")
            records = clean_batch(
                target_spec,
                args.batch_manifest,
                args.out,
                run_label=args.run_label,
                resume=args.resume,
                min_identity=args.min_identity,
                min_raw_coverage=args.min_raw_coverage,
                min_target_coverage=args.min_target_coverage,
            )
        else:
            if not args.models:
                parser.error("--models is required unless --batch-manifest is provided")
            records = clean_models(
                target_spec,
                args.models,
                args.out,
                resume=args.resume,
                min_identity=args.min_identity,
                min_raw_coverage=args.min_raw_coverage,
                min_target_coverage=args.min_target_coverage,
            )
        print(f"cleaned {len(records)} model(s)")
        return 0

    if args.command == "merge-clean":
        rows = merge_clean_batch_manifests(args.out, target_prefix=args.target_prefix)
        print(f"merged {len(rows)} cleaned model row(s)")
        return 0

    if args.command == "af2rank":
        from .af2rank_runner import run_af2rank_stage

        target_spec = load_target_spec_json(args.target_spec)
        records = run_af2rank_stage(target_spec, args.out, _af2rank_config(args), resume=args.resume)
        print(f"af2rank completed {len(records)} model(s)")
        return 0

    if args.command == "dockq-io":
        from .dockq_io import run_dockq_io_stage

        target_spec = load_target_spec_json(args.target_spec)
        records = run_dockq_io_stage(
            target_spec,
            args.out,
            dockq_src=args.dockq_src,
            jobs=args.jobs,
        )
        print(f"dockq-io completed {len(records)} model(s)")
        return 0

    if args.command == "aggregate":
        from .aggregate import aggregate_results

        rows = aggregate_results(args.out, allow_partial=args.allow_partial)
        print(f"aggregated {len(rows)} model(s)")
        return 0

    if args.command == "merge-af2rank":
        from .af2rank_runner import merge_af2rank_batch_manifests

        af2rank_rows, dockq_rows = merge_af2rank_batch_manifests(args.out)
        print(f"merged {len(af2rank_rows)} af2rank row(s)")
        print(f"merged {len(dockq_rows)} dockq-pae manifest row(s)")
        return 0

    if args.command == "make-batches":
        from .batching import prepare_raw_batches

        count = prepare_raw_batches(
            args.target,
            args.fasta,
            args.models,
            args.out,
            args.batch_root,
            num_batches=args.num_batches,
            overwrite=args.overwrite,
        )
        print(f"created {count} batch(es)")
        return 0

    if args.command == "materialize-af2rank-batch":
        from .batching import materialize_cleaned_batch

        target_spec = load_target_spec_json(args.target_spec)
        count = materialize_cleaned_batch(target_spec.target, args.out, args.batch_manifest, args.dest_dir)
        print(f"materialized {count} cleaned PDB(s)")
        return 0

    if args.command == "batch-cleanup":
        from .cleanup import cleanup_af2rank_batch

        summary = cleanup_af2rank_batch(args.out, args.target_prefix, args.batch_id)
        print(f"removed raw_npz dirs: {len(summary['removed_raw_npz'])}")
        print(f"compressed pae jsons: {len(summary['compressed_pae_json'])}")
        return 0

    if args.command == "cleanup":
        from .cleanup import cleanup_outputs

        summary = cleanup_outputs(
            args.out,
            archive=args.archive,
            remove=not args.no_remove,
            include_pae_json=args.remove_pae_json,
            include_scored_pdbs=args.include_scored_pdbs,
            gzip_pae_json=not args.no_gzip_pae_json,
            archive_name=args.archive_name,
        )
        print(f"heavy paths found: {len(summary['paths'])}")
        if summary["archive_written"]:
            print(f"archive: {summary['archive']}")
        if summary["compressed_pae_json"]:
            print(f"compressed pae jsons: {len(summary['compressed_pae_json'])}")
        if summary["remove"]:
            print(f"removed paths: {len(summary['removed'])}")
        return 0

    if args.command == "run-target":
        from .af2rank_runner import run_af2rank_stage
        from .aggregate import aggregate_results
        from .dockq_io import run_dockq_io_stage

        target_spec = load_target_spec(args.target, args.fasta)
        if not args.models:
            parser.error("--models is required for run-target")
        clean_records = clean_models(
            target_spec,
            args.models,
            args.out,
            resume=args.resume,
            min_identity=args.min_identity,
            min_raw_coverage=args.min_raw_coverage,
            min_target_coverage=args.min_target_coverage,
        )
        print(f"cleaned {len(clean_records)} model(s)")
        af2_records = run_af2rank_stage(target_spec, args.out, _af2rank_config(args), resume=args.resume)
        print(f"af2rank completed {len(af2_records)} model(s)")
        if not args.skip_dockq:
            dockq_records = run_dockq_io_stage(
                target_spec,
                args.out,
                dockq_src=args.dockq_src,
                jobs=args.workers,
            )
            print(f"dockq-io completed {len(dockq_records)} model(s)")
        rows = aggregate_results(
            args.out,
            require_dockq=not args.skip_dockq,
        )
        print(f"aggregated {len(rows)} model(s)")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
