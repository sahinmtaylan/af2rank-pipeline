from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from .exceptions import ChainMappingError, DependencyError
from .structure import ChainRecord
from .target import TargetChain


@dataclass(frozen=True)
class PairwiseChainAlignment:
    raw_chain_id: str
    target_chain_id: str
    raw_sequence: str
    target_sequence: str
    residue_map: dict[int, int]
    matches: int
    mismatches: int
    raw_aligned: int
    target_aligned: int
    identity: float
    raw_coverage: float
    target_coverage: float
    score: float
    method: str

    @property
    def matched_residue_map(self) -> dict[int, int]:
        return {
            raw_idx: target_idx
            for raw_idx, target_idx in self.residue_map.items()
            if self.raw_sequence[raw_idx] == self.target_sequence[target_idx]
        }


def _exact_alignment(raw: ChainRecord, target: TargetChain) -> PairwiseChainAlignment | None:
    if raw.sequence != target.sequence:
        return None
    residue_map = {idx: idx for idx in range(len(raw.sequence))}
    return PairwiseChainAlignment(
        raw_chain_id=raw.chain_id,
        target_chain_id=target.canonical_id,
        raw_sequence=raw.sequence,
        target_sequence=target.sequence,
        residue_map=residue_map,
        matches=len(raw.sequence),
        mismatches=0,
        raw_aligned=len(raw.sequence),
        target_aligned=len(target.sequence),
        identity=1.0,
        raw_coverage=1.0,
        target_coverage=1.0,
        score=1.0,
        method="exact",
    )


def _substring_alignment(raw: ChainRecord, target: TargetChain) -> PairwiseChainAlignment | None:
    if not raw.sequence or raw.sequence not in target.sequence:
        return None
    offset = target.sequence.index(raw.sequence)
    residue_map = {idx: offset + idx for idx in range(len(raw.sequence))}
    raw_cov = 1.0
    target_cov = len(raw.sequence) / len(target.sequence)
    score = 0.55 + 0.30 * target_cov + 0.15 * raw_cov
    return PairwiseChainAlignment(
        raw_chain_id=raw.chain_id,
        target_chain_id=target.canonical_id,
        raw_sequence=raw.sequence,
        target_sequence=target.sequence,
        residue_map=residue_map,
        matches=len(raw.sequence),
        mismatches=0,
        raw_aligned=len(raw.sequence),
        target_aligned=len(raw.sequence),
        identity=1.0,
        raw_coverage=raw_cov,
        target_coverage=target_cov,
        score=score,
        method="substring",
    )


def _biopython_semiglobal_alignment(raw: ChainRecord, target: TargetChain) -> PairwiseChainAlignment:
    """Semi-global alignment via Biopython: terminal gaps free, internal gaps penalized."""
    try:
        from Bio.Align import PairwiseAligner
    except ImportError as exc:
        raise DependencyError("Pairwise chain alignment requires biopython.") from exc

    query = raw.sequence
    ref = target.sequence
    n = len(query)
    m = len(ref)

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -2.0
    aligner.open_gap_score = -3.0
    aligner.extend_gap_score = -3.0
    aligner.query_end_gap_score = 0.0
    aligner.target_end_gap_score = 0.0

    alignment = aligner.align(query, ref)[0]

    residue_map: dict[int, int] = {}
    matches = 0
    mismatches = 0
    raw_aligned = 0
    target_aligned = 0

    query_blocks, ref_blocks = alignment.aligned
    for query_block, ref_block in zip(query_blocks, ref_blocks):
        query_start, query_end = map(int, query_block)
        ref_start, ref_end = map(int, ref_block)
        block_len = min(query_end - query_start, ref_end - ref_start)
        for offset in range(block_len):
            raw_idx = query_start + offset
            target_idx = ref_start + offset
            residue_map[raw_idx] = target_idx
            raw_aligned += 1
            target_aligned += 1
            if query[raw_idx] == ref[target_idx]:
                matches += 1
            else:
                mismatches += 1

    residue_map = dict(sorted(residue_map.items()))
    aligned = max(1, raw_aligned)
    identity = matches / aligned
    raw_coverage = raw_aligned / n if n else 0.0
    target_coverage = target_aligned / m if m else 0.0
    gap_burden = 1.0 - min(raw_coverage, target_coverage)
    score = 0.55 * identity + 0.30 * target_coverage + 0.15 * raw_coverage - 0.10 * gap_burden

    return PairwiseChainAlignment(
        raw_chain_id=raw.chain_id,
        target_chain_id=target.canonical_id,
        raw_sequence=query,
        target_sequence=ref,
        residue_map=residue_map,
        matches=matches,
        mismatches=mismatches,
        raw_aligned=raw_aligned,
        target_aligned=target_aligned,
        identity=identity,
        raw_coverage=raw_coverage,
        target_coverage=target_coverage,
        score=score,
        method="biopython_semiglobal",
    )


def align_chain_to_target(raw: ChainRecord, target: TargetChain) -> PairwiseChainAlignment:
    return (
        _exact_alignment(raw, target)
        or _substring_alignment(raw, target)
        or _biopython_semiglobal_alignment(raw, target)
    )


def _assignment_score(
    assignment: tuple[ChainRecord, ...],
    targets: list[TargetChain],
    matrix: dict[tuple[str, str], PairwiseChainAlignment],
    min_identity: float,
    min_raw_coverage: float,
    min_target_coverage: float,
) -> tuple[float, list[PairwiseChainAlignment]] | None:
    alignments: list[PairwiseChainAlignment] = []
    total = 0.0
    for raw_chain, target in zip(assignment, targets):
        alignment = matrix[(raw_chain.chain_id, target.canonical_id)]
        if alignment.identity < min_identity:
            return None
        if alignment.raw_coverage < min_raw_coverage:
            return None
        if alignment.target_coverage < min_target_coverage:
            return None
        alignments.append(alignment)
        total += alignment.score
    return total, alignments


def assign_chains(
    raw_chains: list[ChainRecord],
    targets: list[TargetChain],
    *,
    min_identity: float = 0.90,
    min_raw_coverage: float = 0.85,
    min_target_coverage: float = 0.50,
    max_permutations: int = 200_000,
) -> list[PairwiseChainAlignment]:
    if len(raw_chains) < len(targets):
        raise ChainMappingError(
            f"Model has {len(raw_chains)} protein chains but target expects {len(targets)} chains"
        )

    matrix = {
        (raw.chain_id, target.canonical_id): align_chain_to_target(raw, target)
        for raw in raw_chains
        for target in targets
    }

    num_permutations = math.factorial(len(raw_chains)) // math.factorial(len(raw_chains) - len(targets))
    if num_permutations > max_permutations:
        return _assign_chains_hungarian(
            raw_chains,
            targets,
            matrix,
            min_identity=min_identity,
            min_raw_coverage=min_raw_coverage,
            min_target_coverage=min_target_coverage,
        )

    best: tuple[float, list[PairwiseChainAlignment]] | None = None
    for assignment in itertools.permutations(raw_chains, len(targets)):
        scored = _assignment_score(
            assignment,
            targets,
            matrix,
            min_identity=min_identity,
            min_raw_coverage=min_raw_coverage,
            min_target_coverage=min_target_coverage,
        )
        if scored is not None and (best is None or scored[0] > best[0]):
            best = scored

    if best is None:
        diagnostics = []
        for target in targets:
            candidates = sorted(
                (matrix[(raw.chain_id, target.canonical_id)] for raw in raw_chains),
                key=lambda item: item.score,
                reverse=True,
            )[:3]
            diagnostics.append(
                f"{target.canonical_id}: "
                + ", ".join(
                    f"{candidate.raw_chain_id} id={candidate.identity:.3f} raw_cov={candidate.raw_coverage:.3f} target_cov={candidate.target_coverage:.3f}"
                    for candidate in candidates
                )
            )
        raise ChainMappingError("No valid chain assignment. Best candidates: " + " | ".join(diagnostics))

    return best[1]


def _assign_chains_hungarian(
    raw_chains: list[ChainRecord],
    targets: list[TargetChain],
    matrix: dict[tuple[str, str], PairwiseChainAlignment],
    *,
    min_identity: float,
    min_raw_coverage: float,
    min_target_coverage: float,
) -> list[PairwiseChainAlignment]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise DependencyError(
            "Large chain assignments require scipy. Install the 'many-chains' extra or lower chain count."
        ) from exc

    cost_matrix = []
    for target in targets:
        row = []
        for raw in raw_chains:
            alignment = matrix[(raw.chain_id, target.canonical_id)]
            row.append(-alignment.score)
        cost_matrix.append(row)

    target_indices, raw_indices = linear_sum_assignment(cost_matrix)
    alignments = []
    for target_idx, raw_idx in zip(target_indices, raw_indices):
        target = targets[target_idx]
        raw = raw_chains[raw_idx]
        alignment = matrix[(raw.chain_id, target.canonical_id)]
        if alignment.identity < min_identity:
            raise ChainMappingError("Hungarian assignment failed identity threshold")
        if alignment.raw_coverage < min_raw_coverage or alignment.target_coverage < min_target_coverage:
            raise ChainMappingError("Hungarian assignment failed coverage threshold")
        alignments.append(alignment)

    return sorted(alignments, key=lambda item: item.target_chain_id)
