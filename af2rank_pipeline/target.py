from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

from .exceptions import PipelineError

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
CANONICAL_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


@dataclass(frozen=True)
class TargetChain:
    canonical_id: str
    header: str
    sequence: str
    length: int
    offset: int
    subunit: str | None = None


@dataclass(frozen=True)
class TargetSpec:
    target: str
    fasta_path: str
    chains: List[TargetChain]

    @property
    def flat_sequence(self) -> str:
        return "".join(chain.sequence for chain in self.chains)

    @property
    def chain_ids(self) -> str:
        return "".join(chain.canonical_id for chain in self.chains)

    @property
    def length(self) -> int:
        return len(self.flat_sequence)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "fasta_path": self.fasta_path,
            "length": self.length,
            "flat_sequence": self.flat_sequence,
            "chain_ids": self.chain_ids,
            "chains": [asdict(chain) for chain in self.chains],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def parse_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_parts).upper()))
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(re.sub(r"\s+", "", line))

    if header is not None:
        records.append((header, "".join(seq_parts).upper()))
    return records


def load_target_spec(target: str, fasta_path: str | Path) -> TargetSpec:
    path = Path(fasta_path).expanduser().resolve()
    if not path.exists():
        raise PipelineError(f"FASTA file does not exist: {path}")

    records = parse_fasta_records(path)
    if not records:
        raise PipelineError(f"No FASTA records found in {path}")
    if len(records) > len(CANONICAL_CHAIN_IDS):
        raise PipelineError(
            f"{path} has {len(records)} chains; only {len(CANONICAL_CHAIN_IDS)} canonical chain IDs are supported"
        )

    seen_headers: set[str] = set()
    chains: list[TargetChain] = []
    offset = 0
    for idx, (header, sequence) in enumerate(records):
        if not sequence:
            raise PipelineError(f"FASTA record {idx + 1} has an empty sequence")
        if header in seen_headers:
            raise PipelineError(f"Duplicate FASTA header is ambiguous: {header}")
        seen_headers.add(header)

        unsupported = sorted(set(sequence) - STANDARD_AA)
        if unsupported:
            raise PipelineError(
                f"FASTA record {idx + 1} contains unsupported residues: {','.join(unsupported)}"
            )

        subunit_match = re.search(r"\bsubunit\s+([^,;]+)", header, re.IGNORECASE)
        chain = TargetChain(
            canonical_id=CANONICAL_CHAIN_IDS[idx],
            header=header,
            sequence=sequence,
            length=len(sequence),
            offset=offset,
            subunit=subunit_match.group(1).strip() if subunit_match else None,
        )
        chains.append(chain)
        offset += len(sequence)

    return TargetSpec(target=target, fasta_path=str(path), chains=chains)


def target_from_dict(data: dict) -> TargetSpec:
    return TargetSpec(
        target=data["target"],
        fasta_path=data["fasta_path"],
        chains=[TargetChain(**chain) for chain in data["chains"]],
    )


def load_target_spec_json(path: str | Path) -> TargetSpec:
    return target_from_dict(json.loads(Path(path).read_text()))
