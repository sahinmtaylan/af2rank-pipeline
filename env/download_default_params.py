"""Download the default AF2Rank weight from DeepMind's dated AlphaFold archive.

The archive is about 5.3 GB; this helper uses HTTP byte ranges to fetch only
params_model_2_ptm.npz. No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


ARCHIVE_URL = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
WEIGHT_NAME = "params_model_2_ptm.npz"
WEIGHT_SHA256 = "23645d9a82c4af2ed54cd48a7b3c1c2575dc6aa9fe931adb4d7203ca5f0dc398"


def get_range(start: int, end: int):
    request = urllib.request.Request(ARCHIVE_URL, headers={"Range": f"bytes={start}-{end}"})
    response = urllib.request.urlopen(request, timeout=60)
    expected = f"bytes {start}-{end}/"
    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
        response.close()
        raise RuntimeError("The AlphaFold archive server did not honor the byte-range request")
    return response


def locate_weight() -> tuple[int, int]:
    offset = 0
    for _ in range(100):
        with get_range(offset, offset + 511) as response:
            header = response.read()
        if len(header) != 512 or header[257:262] != b"ustar":
            raise RuntimeError(f"Invalid archive header at byte {offset}")
        name = header[:100].split(b"\0", 1)[0].decode("ascii")
        size_field = header[124:136].split(b"\0", 1)[0].strip()
        size = int(size_field, 8)
        if name == WEIGHT_NAME:
            return offset + 512, size
        offset += 512 + ((size + 511) // 512) * 512
    raise RuntimeError(f"{WEIGHT_NAME} was not found in the official archive")


def verify_weight(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise RuntimeError(f"Unexpected checksum for {path}; remove the file and retry")
    with zipfile.ZipFile(path) as archive:
        corrupt_member = archive.testzip()
    if corrupt_member is not None:
        raise RuntimeError(f"Corrupt parameter archive member: {corrupt_member}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("params_dir", type=Path, help="Destination params directory")
    args = parser.parse_args()
    params_dir = args.params_dir.expanduser().resolve()
    params_dir.mkdir(parents=True, exist_ok=True)
    destination = params_dir / WEIGHT_NAME
    if destination.is_file():
        verify_weight(destination)
        print(f"Verified existing weight: {destination}")
        return

    start, size = locate_weight()
    partial = destination.with_suffix(".npz.part")
    try:
        with get_range(start, start + size - 1) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if partial.stat().st_size != size:
            raise RuntimeError(f"Incomplete download: expected {size} bytes")
        verify_weight(partial)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Downloaded and verified: {destination}")


if __name__ == "__main__":
    main()
