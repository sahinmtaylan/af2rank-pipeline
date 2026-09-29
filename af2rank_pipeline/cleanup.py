from __future__ import annotations

import csv
import gzip
import json
import shutil
import tarfile
from pathlib import Path

from .manifests import ensure_run_dirs


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _path_has_entries(path: Path) -> bool:
    if path.is_file():
        return True
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    return True


def discover_heavy_outputs(
    out_dir: str | Path,
    *,
    include_raw_npz: bool = True,
    include_pae_json: bool = False,
    include_scored_pdbs: bool = False,
) -> list[Path]:
    dirs = ensure_run_dirs(out_dir)
    root = dirs["root"]
    candidates: list[Path] = []
    if include_raw_npz:
        candidates.extend(path for path in dirs["af2rank"].rglob("raw_npz") if path.is_dir())
    if include_scored_pdbs:
        candidates.extend(path for path in dirs["af2rank"].rglob("scored_pdbs") if path.is_dir())
    if include_pae_json and dirs["af2rank_pae"].is_dir():
        candidates.append(dirs["af2rank_pae"])

    unique: list[Path] = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not _is_inside(path, root) or not _path_has_entries(path):
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique)


def _gzip_file(path: Path, gz_path: Path) -> None:
    tmp_path = gz_path.with_name(f".{gz_path.name}.tmp")
    gz_path.parent.mkdir(parents=True, exist_ok=True)
    if gz_path.exists():
        path.unlink(missing_ok=True)
        return
    with path.open("rb") as src, gzip.open(tmp_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp_path.replace(gz_path)
    path.unlink()


def gzip_pae_jsons(out_dir: str | Path) -> list[str]:
    dirs = ensure_run_dirs(out_dir)
    compressed = []
    if not dirs["af2rank_pae"].is_dir():
        return compressed
    for path in sorted(dirs["af2rank_pae"].glob("*.json")):
        gz_path = path.with_suffix(path.suffix + ".gz")
        _gzip_file(path, gz_path)
        compressed.append(str(gz_path))
    for manifest in sorted(dirs["manifests"].glob("af2rank*.csv")):
        compressed.extend(_gzip_manifest_pae_paths(manifest, ["pae_json"]))
    for manifest in sorted(dirs["manifests"].glob("dockq_pae_manifest*.csv")):
        compressed.extend(_gzip_manifest_pae_paths(manifest, ["pae"]))
    return sorted(set(compressed))


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        return [], []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    if not fieldnames:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _gzip_manifest_pae_paths(path: Path, columns: list[str]) -> list[str]:
    rows, fieldnames = _read_csv(path)
    if not rows:
        return []
    compressed = []
    changed = False
    for row in rows:
        for column in columns:
            value = row.get(column, "")
            if not value or value.endswith(".gz"):
                continue
            pae_path = Path(value)
            if pae_path.suffix != ".json":
                continue
            gz_path = pae_path.with_suffix(pae_path.suffix + ".gz")
            if pae_path.is_file():
                _gzip_file(pae_path, gz_path)
                compressed.append(str(gz_path))
            elif not gz_path.is_file():
                continue
            row[column] = str(gz_path)
            changed = True
    if changed:
        _write_csv(path, rows, fieldnames)
    return compressed


def cleanup_af2rank_batch(out_dir: str | Path, target_prefix: str, batch_id: str) -> dict:
    dirs = ensure_run_dirs(out_dir)
    batch_id = f"{int(batch_id):03d}" if str(batch_id).isdigit() else str(batch_id)
    run_label = f"{target_prefix}_batch_{batch_id}"
    batch_dir = dirs["af2rank"] / run_label
    removed_raw_npz = []
    for raw_dir in sorted(batch_dir.rglob("raw_npz")):
        if raw_dir.is_dir():
            shutil.rmtree(raw_dir)
            removed_raw_npz.append(str(raw_dir))

    compressed = []
    compressed.extend(_gzip_manifest_pae_paths(dirs["manifests"] / f"af2rank_{run_label}.csv", ["pae_json"]))
    compressed.extend(_gzip_manifest_pae_paths(dirs["manifests"] / f"dockq_pae_manifest_{run_label}.csv", ["pae"]))

    summary = {
        "out_dir": str(dirs["root"]),
        "run_label": run_label,
        "removed_raw_npz": removed_raw_npz,
        "compressed_pae_json": sorted(set(compressed)),
    }
    (dirs["manifests"] / f"cleanup_{run_label}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def cleanup_outputs(
    out_dir: str | Path,
    *,
    archive: bool = False,
    remove: bool = True,
    include_raw_npz: bool = True,
    include_pae_json: bool = False,
    include_scored_pdbs: bool = False,
    gzip_pae_json: bool = True,
    archive_name: str = "heavy_intermediates.tar.gz",
) -> dict:
    dirs = ensure_run_dirs(out_dir)
    root = dirs["root"]
    compressed = gzip_pae_jsons(root) if gzip_pae_json else []
    paths = discover_heavy_outputs(
        root,
        include_raw_npz=include_raw_npz,
        include_pae_json=include_pae_json,
        include_scored_pdbs=include_scored_pdbs,
    )
    archives_dir = root / "archives"
    archive_path = archives_dir / archive_name

    archived = []
    if archive and paths:
        archives_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as tar:
            for path in paths:
                tar.add(path, arcname=path.relative_to(root))
                archived.append(str(path))

    removed = []
    if remove:
        for path in paths:
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(str(path))

    summary = {
        "out_dir": str(root),
        "archive": str(archive_path) if archive and paths else "",
        "archive_written": bool(archive and paths),
        "remove": remove,
        "include_raw_npz": include_raw_npz,
        "include_pae_json": include_pae_json,
        "include_scored_pdbs": include_scored_pdbs,
        "gzip_pae_json": gzip_pae_json,
        "compressed_pae_json": compressed,
        "paths": [str(path) for path in paths],
        "archived": archived,
        "removed": removed,
    }
    (dirs["manifests"] / "cleanup_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
