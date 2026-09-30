#!/usr/bin/env python3
"""AF2Rank inference and actifpTM metrics for cleaned protein-complex models.

Uses installed ColabDesign and AlphaFold parameters supplied at runtime. Writes confidence
metrics, scored structures and PAE data. Raw debug arrays are optional.
"""

import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np


ACTIFPTM_VARIANT = "both_probability_weighted_and_binary_contacts"


def positive_iterations(value):
    try:
        iterations = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("iterations must be an integer of at least 1") from exc
    if iterations < 1:
        raise argparse.ArgumentTypeError("iterations must be at least 1")
    return iterations


def parse_args():
    parser = argparse.ArgumentParser(description="Run AF2Rank and ActifPTM cleanly.")
    parser.add_argument("--decoy_dir", required=True,
                        help="Directory containing input/decoy PDB files.")
    parser.add_argument("--chain", default=None, help="Comma-separated chain IDs, for example A,B,C.")
    parser.add_argument("--model_mode", default="alphafold", choices=["alphafold", "alphafold-multimer"])
    parser.add_argument("--model_num", type=int, default=2)
    parser.add_argument("--recycle", type=int, default=1)
    parser.add_argument("--iterations", type=positive_iterations, default=1)
    parser.add_argument("--params", required=True, help="AlphaFold params dir or parent data dir.")
    parser.add_argument("--colabdesign", default=None, help="Optional ColabDesign checkout override.")
    parser.add_argument("--tm", default=None, help="Optional TMscore executable; if absent, TM-score metrics are skipped.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where final CSV/JSON/PDB outputs are written.")
    parser.add_argument("--pae_json_dir", default=None,
                        help="Directory where per-model PAE JSON files are written.")
    parser.add_argument("--no_pae_json_gzip", action="store_false", dest="pae_json_gzip",
                        help="Write PAE JSON without gzip compression.")
    parser.add_argument("--target", default="target")
    parser.add_argument("--mask_sequence", action="store_true")
    parser.add_argument("--mask_sidechains", action="store_true")
    parser.add_argument("--mask_interchain", action="store_true")
    parser.add_argument("--keep_raw", action="store_true", help="Save raw PAE/distogram NPZ files for debugging.")
    parser.set_defaults(pae_json_gzip=True)
    return parser.parse_args()


def setup_imports(colabdesign_dir):
    if colabdesign_dir:
        checkout = Path(colabdesign_dir).expanduser().resolve()
        if not (checkout / "colabdesign").is_dir():
            raise FileNotFoundError(f"ColabDesign checkout not found: {checkout}")
        if str(checkout) not in sys.path:
            sys.path.insert(0, str(checkout))
    try:
        import colabdesign  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ColabDesign is unavailable. Install the inference extra with "
            "'uv sync --locked --all-extras', or pass --colabdesign to a checkout."
        ) from exc


def get_data_dir(params_path):
    """ColabDesign expects data_dir/params/*.npz; users often pass the params dir itself."""
    p = Path(params_path).resolve()
    if p.name == "params":
        return str(p.parent)
    return str(p)


def model_name_from_args(model_mode, model_num):
    if model_mode == "alphafold":
        return f"model_{model_num}_ptm"
    return f"model_{model_num}_multimer_v3"


def validate_params(params_path, model_mode, model_num):
    """Check that ColabDesign can read the requested params directly from shared storage."""
    params_path = Path(params_path).resolve()
    params_dir = params_path if params_path.name == "params" else params_path / "params"
    if not params_dir.exists():
        raise FileNotFoundError(f"Parameters directory not found: {params_dir}")

    if model_mode == "alphafold":
        expected = params_dir / f"params_model_{model_num}_ptm.npz"
    else:
        expected = params_dir / f"params_model_{model_num}_multimer_v3.npz"

    if not expected.exists():
        raise FileNotFoundError(f"Expected model parameters not found: {expected}")


def parse_chain_labels(chain):
    if chain is None or chain.strip() == "":
        return None
    return [x.strip() for x in chain.split(",") if x.strip()]


def get_chain_lengths_from_pdb(pdb_path, chain_labels=None):
    """Return chain lengths from ATOM records, using unique residues per chain."""
    chain_order = []
    residues_by_chain = {}
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21].strip() or "_"
            if chain_labels is not None and chain_id not in chain_labels:
                continue
            res_key = (line[22:26].strip(), line[26].strip())
            if chain_id not in residues_by_chain:
                residues_by_chain[chain_id] = set()
                chain_order.append(chain_id)
            residues_by_chain[chain_id].add(res_key)

    if chain_labels is not None:
        chain_order = [c for c in chain_labels if c in residues_by_chain]

    return [len(residues_by_chain[c]) for c in chain_order], chain_order


def build_asym_id_from_lengths(lengths):
    if not lengths:
        return None
    return np.concatenate([np.full(length, i, dtype=np.int32) for i, length in enumerate(lengths)])


def get_asym_id(af_model, pdb_path, chain):
    inputs = af_model.model._inputs
    batch = inputs.get("batch", {})
    # Chain IDs must follow prepared residues rather than raw PDB counts.
    for features in (inputs, batch):
        if "asym_id" in features:
            asym_id = np.array(features["asym_id"])
            return asym_id[0] if asym_id.ndim > 1 else asym_id

    if "res_mask_per_chain" in af_model.model._inputs:
        masks = np.array(af_model.model._inputs["res_mask_per_chain"])
        asym_id = np.zeros(masks.shape[1], dtype=np.int32)
        for i in range(masks.shape[0]):
            asym_id[masks[i].astype(bool)] = i
        return asym_id

    labels = parse_chain_labels(chain)
    lengths, _ = get_chain_lengths_from_pdb(pdb_path, labels)
    asym_id = build_asym_id_from_lengths(lengths)
    if asym_id is not None:
        return asym_id

    L = len(batch["aatype"])
    return np.zeros(L, dtype=np.int32)


def chain_ranges_from_asym_id(asym_id, chain_labels=None):
    asym_id = np.array(asym_id)
    if asym_id.ndim > 1:
        asym_id = asym_id[0]
    unique = list(np.unique(asym_id))
    if chain_labels is None:
        chain_labels = [chr(ord("A") + i) for i in range(len(unique))]
    ranges = {}
    for i, chain_value in enumerate(unique):
        positions = np.where(asym_id == chain_value)[0]
        if len(positions) == 0:
            continue
        label = chain_labels[i] if i < len(chain_labels) else f"chain{i + 1}"
        ranges[label] = (int(positions[0]), int(positions[-1]))
    return ranges


def calculate_bin_centers(breaks, use_jnp=False):
    import jax.numpy as jnp

    xp = jnp if use_jnp else np
    breaks = xp.array(breaks)
    return 0.5 * (breaks[:-1] + breaks[1:])


def softmax(x, axis=-1):
    import scipy.special

    return scipy.special.softmax(x, axis=axis)


def predicted_tm_score_modified(logits, breaks, residue_weights=None, asym_id=None,
                                pair_residue_weights=None, use_jnp=False):
    import jax.numpy as jnp

    xp = jnp if use_jnp else np
    logits = xp.array(logits)
    breaks = xp.array(breaks)

    if residue_weights is None:
        residue_weights = xp.ones(logits.shape[0])
    else:
        residue_weights = xp.array(residue_weights)

    bin_centers = calculate_bin_centers(breaks, use_jnp=use_jnp)
    if bin_centers.shape[0] != logits.shape[-1]:
        recreated_breaks = xp.linspace(0, 31, logits.shape[-1] + 1)
        bin_centers = calculate_bin_centers(recreated_breaks, use_jnp=use_jnp)

    num_res = residue_weights.shape[0]
    clipped_num_res = xp.maximum(num_res, 19)
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8
    tm_per_bin = 1.0 / (1 + xp.square(bin_centers) / xp.square(d0))

    probs = jnp.asarray(softmax(np.asarray(logits), axis=-1)) if use_jnp else softmax(logits, axis=-1)
    predicted_tm_term = (probs * tm_per_bin).sum(-1)

    if asym_id is None:
        pair_mask = xp.ones(predicted_tm_term.shape, dtype=bool)
    else:
        asym_id = xp.array(asym_id)
        pair_mask = asym_id[:, None] != asym_id[None, :]

    predicted_tm_term *= pair_mask

    if pair_residue_weights is None:
        pair_residue_weights = pair_mask * (residue_weights[None, :] * residue_weights[:, None])
    else:
        pair_residue_weights = xp.array(pair_residue_weights)

    normed_residue_mask = pair_residue_weights / (1e-8 + pair_residue_weights.sum(-1, keepdims=True))
    per_alignment = (predicted_tm_term * normed_residue_mask).sum(-1)
    return per_alignment * residue_weights


def distogram_bins(distogram):
    logits = np.array(distogram["logits"])
    bin_edges = np.array(distogram["bin_edges"])

    if logits.shape[-1] == 64:
        if bin_edges.shape[0] == 63:
            return np.append(0, bin_edges)
        return np.append(0, np.linspace(2.3125, 21.6875, 63))
    if logits.shape[-1] == 39:
        return np.linspace(3.25, 50.75, 39) + 1.25
    if bin_edges.shape[0] == logits.shape[-1]:
        return bin_edges

    return np.linspace(0, 31, logits.shape[-1])


def contact_map(distogram, dist=8.0):
    logits = np.array(distogram["logits"])
    bins = distogram_bins(distogram)
    return (softmax(logits, axis=-1) * (bins < dist)).sum(-1)


def compute_pair_metrics(outputs, asym_id, chain_labels=None):
    import jax.numpy as jnp

    asym_id = np.array(asym_id)
    if asym_id.ndim > 1:
        asym_id = asym_id[0]

    chain_ranges = chain_ranges_from_asym_id(asym_id, chain_labels)
    distogram = outputs["distogram"]
    pae = outputs["predicted_aligned_error"]
    cmap = contact_map(distogram)

    metrics = {
        "pairwise_actifptm": {},
        "pairwise_actifptm_binary": {},
        "pairwise_iptm": {},
        "per_chain_ptm": {},
        "actifptm": None,
        "actifptm_prob": None,
        "actifptm_binary": None,
    }

    full_length = len(asym_id)
    breaks = pae["breaks"]
    logits = pae["logits"]

    for label, (start, end) in chain_ranges.items():
        sliced_logits = jnp.array(logits)[start:end + 1, start:end + 1, :]
        residue_weights = jnp.ones(sliced_logits.shape[0], dtype=float)
        score = predicted_tm_score_modified(
            sliced_logits, breaks, residue_weights=residue_weights, use_jnp=True
        )
        metrics["per_chain_ptm"][label] = round(float(score.max()), 3)

    labels = list(chain_ranges.keys())
    full_prob_weights = np.zeros((full_length, full_length), dtype=float)
    full_binary_weights = np.zeros((full_length, full_length), dtype=float)

    for i, label_i in enumerate(labels):
        start_i, end_i = chain_ranges[label_i]
        for j, label_j in enumerate(labels):
            if j <= i:
                continue
            start_j, end_j = chain_ranges[label_j]
            key = f"{label_i}_{label_j}"

            seq_mask = np.zeros(full_length, dtype=float)
            seq_mask[start_i:end_i + 1] = 1.0
            seq_mask[start_j:end_j + 1] = 1.0

            pair_prob_weights = np.zeros((full_length, full_length), dtype=float)
            pair_prob_weights[start_i:end_i + 1, start_j:end_j + 1] = cmap[start_i:end_i + 1, start_j:end_j + 1]
            pair_prob_weights[start_j:end_j + 1, start_i:end_i + 1] = cmap[start_j:end_j + 1, start_i:end_i + 1]
            full_prob_weights += pair_prob_weights

            actif_prob = predicted_tm_score_modified(
                logits, breaks, residue_weights=seq_mask, asym_id=asym_id,
                pair_residue_weights=pair_prob_weights, use_jnp=True
            )
            metrics["pairwise_actifptm"][key] = round(float(actif_prob.max()), 3)

            contacts = np.where(cmap[start_i:end_i + 1, start_j:end_j + 1] >= 0.6)
            contact_seq_mask = np.zeros(full_length, dtype=float)
            if contacts[0].size > 0:
                global_i = contacts[0] + start_i
                global_j = contacts[1] + start_j
                contact_positions = np.unique(np.concatenate([global_i, global_j]))
                contact_seq_mask[contact_positions] = 1.0
                actif_binary = predicted_tm_score_modified(
                    logits, breaks, residue_weights=contact_seq_mask, asym_id=asym_id, use_jnp=True
                )
                metrics["pairwise_actifptm_binary"][key] = round(float(actif_binary.max()), 3)
                full_binary_weights += contact_seq_mask[None, :] * contact_seq_mask[:, None]
            else:
                metrics["pairwise_actifptm_binary"][key] = 0.0

            iptm = predicted_tm_score_modified(
                logits, breaks, residue_weights=seq_mask, asym_id=asym_id, use_jnp=True
            )
            metrics["pairwise_iptm"][key] = round(float(iptm.max()), 3)

    if len(labels) > 1:
        residue_weights = np.ones(full_length, dtype=float)
        actif_prob = predicted_tm_score_modified(
            logits, breaks, residue_weights=residue_weights, asym_id=asym_id,
            pair_residue_weights=full_prob_weights, use_jnp=True
        )
        pair_mask = asym_id[:, None] != asym_id[None, :]
        full_binary_weights *= pair_mask
        actif_binary = predicted_tm_score_modified(
            logits, breaks, residue_weights=residue_weights, asym_id=asym_id,
            pair_residue_weights=full_binary_weights, use_jnp=True
        )
        metrics["actifptm"] = round(float(actif_prob.max()), 3)
        metrics["actifptm_prob"] = metrics["actifptm"]
        metrics["actifptm_binary"] = round(float(actif_binary.max()), 3)
    else:
        metrics["actifptm"] = 0.0
        metrics["actifptm_prob"] = 0.0
        metrics["actifptm_binary"] = 0.0

    return metrics


def validate_tm_executable(tm_executable):
    if not tm_executable:
        return None
    path = Path(tm_executable).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TMscore executable not found: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"TMscore executable is not executable: {path}")
    return str(path)


def tmscore(xyz_a, xyz_b, tm_executable):
    tm_executable = validate_tm_executable(tm_executable)
    if tm_executable is None:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_a = os.path.join(tmpdir, "a.pdb")
        pdb_b = os.path.join(tmpdir, "b.pdb")
        write_ca_pdb(xyz_a, pdb_a)
        write_ca_pdb(xyz_b, pdb_b)
        proc = subprocess.run(
            [tm_executable, pdb_a, pdb_b],
            check=False,
            capture_output=True,
            text=True,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"TMscore exited with status {proc.returncode}: {proc.stderr.strip()[:200]}"
        )

    for line in proc.stdout.splitlines():
        if line.startswith("TM-score"):
            try:
                return float(line.split()[2])
            except (IndexError, ValueError):
                break
    raise ValueError("TMscore completed without a parseable TM-score value")


def write_ca_pdb(xyz, filename):
    with open(filename, "w", encoding="utf-8") as handle:
        for i, (x, y, z) in enumerate(np.array(xyz), start=1):
            handle.write(
                f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        handle.write("TER\nEND\n")


class AF2RankRunner:
    def __init__(self, pdb, chain, model_name, data_dir):
        from colabdesign import mk_af_model
        from colabdesign.shared.utils import copy_dict

        self.copy_dict = copy_dict
        self.args = {
            "pdb": pdb,
            "chain": chain,
            "model_name": model_name,
            "use_multimer": "multimer" in model_name,
        }
        self.model = mk_af_model(
            protocol="fixbb",
            use_templates=True,
            use_multimer=self.args["use_multimer"],
            debug=True,
            model_names=[model_name],
            data_dir=data_dir,
        )
        self.model.prep_inputs(pdb, chain=chain)
        self.model.set_seq(mode="wildtype")
        self.wt_batch = copy_dict(self.model._inputs["batch"])
        self.wt = self.model._wt_aatype

    def set_pdb(self, pdb, chain):
        self.model.prep_inputs(pdb, chain=chain)
        self.model.set_seq(mode="wildtype")
        self.wt = self.model._wt_aatype

    def predict(self, pdb, chain, recycles, iterations, rm_seq, rm_sc, rm_ic, output_pdb=None):
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if pdb is not None:
            self.set_pdb(pdb, chain)

        self.model._inputs["batch"]["aatype"] = self.wt
        self.model.set_opt(template=dict(rm_ic=rm_ic), num_recycles=recycles)
        self.model._inputs["rm_template"][:] = False
        self.model._inputs["rm_template_sc"][:] = rm_sc
        self.model._inputs["rm_template_seq"][:] = rm_seq

        initial_atoms = self.model._inputs["batch"]["all_atom_positions"].copy()
        for i in range(iterations):
            self.model.predict(models=self.args["model_name"], verbose=False)
            if i < iterations - 1:
                self.model._inputs["batch"]["all_atom_positions"] = self.model.aux["atom_positions"]
            else:
                self.model._inputs["batch"]["all_atom_positions"] = initial_atoms

        if output_pdb is not None:
            self.model.save_pdb(output_pdb)

        return self.model.aux


def flatten_metrics(row):
    out = {}
    for key, value in row.items():
        if key in {"pairwise_actifptm", "pairwise_actifptm_binary", "pairwise_iptm", "per_chain_ptm"}:
            continue
        out[key] = value

    for key, value in row.get("pairwise_actifptm", {}).items():
        out[f"actifptm_{key}"] = value
    for key, value in row.get("pairwise_actifptm_binary", {}).items():
        out[f"actifptm_binary_{key}"] = value
    for key, value in row.get("pairwise_iptm", {}).items():
        out[f"iptm_{key}"] = value
    for key, value in row.get("per_chain_ptm", {}).items():
        out[f"chain_ptm_{key}"] = value
    return out


def write_csv(rows, filename):
    flattened = [flatten_metrics(row) for row in rows]
    fieldnames = sorted({key for row in flattened for key in row.keys()})
    with open(filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="N/A")
        writer.writeheader()
        for row in flattened:
            writer.writerow(row)


def maybe_save_raw_npz(out_dir, pdb_id, aux, asym_id):
    outputs = aux["debug"]["outputs"]
    raw_dir = Path(out_dir) / "raw_npz"
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_dir / f"{pdb_id}_raw_outputs.npz",
        pae=np.array(aux.get("pae")),
        pae_logits=np.array(outputs["predicted_aligned_error"]["logits"]),
        pae_breaks=np.array(outputs["predicted_aligned_error"]["breaks"]),
        distogram_logits=np.array(outputs["distogram"]["logits"]),
        distogram_bins=np.array(outputs["distogram"]["bin_edges"]),
        asym_id=np.array(asym_id),
    )


def save_pae_json(pae_json_dir, pdb_id, aux, gzip_enabled=True):
    pae_json_dir = Path(pae_json_dir)
    pae_json_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".json.gz" if gzip_enabled else ".json"
    path = pae_json_dir / f"{pdb_id}{suffix}"
    tmp_path = path.with_name(f".{path.name}.tmp")
    payload = {"model_id": pdb_id, "pae": np.asarray(aux.get("pae"), dtype=float).tolist()}
    if gzip_enabled:
        with gzip.open(tmp_path, "wt") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.write("\n")
    else:
        with tmp_path.open("w") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.write("\n")
    tmp_path.replace(path)
    return path


def main():
    args = parse_args()
    args.tm = validate_tm_executable(args.tm)
    setup_imports(args.colabdesign)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pae_json_dir = Path(args.pae_json_dir) if args.pae_json_dir else out_dir / "pae_json"
    pdb_out_dir = out_dir / "scored_pdbs"
    pdb_out_dir.mkdir(exist_ok=True)

    version = "v3" if args.model_mode == "alphafold-multimer" else "ptm"
    model_name = model_name_from_args(args.model_mode, args.model_num)
    validate_params(args.params, args.model_mode, args.model_num)
    data_dir = get_data_dir(args.params)
    chain_labels = parse_chain_labels(args.chain)

    pdb_files = sorted(Path(args.decoy_dir).glob("*.pdb"))
    if not pdb_files:
        raise SystemExit(f"No .pdb files found in {args.decoy_dir}")

    rows = []
    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem
        print(f"[{args.target}] scoring {pdb_path.name} with {model_name}", flush=True)

        output_pdb = str(pdb_out_dir / f"{pdb_id}_scored.pdb")
        runner = AF2RankRunner(str(pdb_path), args.chain, model_name, data_dir)
        aux = runner.predict(
            str(pdb_path),
            args.chain,
            recycles=args.recycle,
            iterations=args.iterations,
            rm_seq=args.mask_sequence,
            rm_sc=args.mask_sidechains,
            rm_ic=args.mask_interchain,
            output_pdb=output_pdb,
        )

        asym_id = get_asym_id(runner, str(pdb_path), args.chain)
        outputs = aux["debug"]["outputs"]
        actif_metrics = compute_pair_metrics(outputs, asym_id, chain_labels)

        log = deepcopy(aux["log"])
        plddt = float(log.get("plddt", np.nan))
        ptm = float(log.get("ptm", np.nan))
        iptm = float(log.get("i_ptm", np.nan)) if "i_ptm" in log else np.nan

        input_ca = runner.model._inputs["batch"]["all_atom_positions"][:, 1]
        output_ca = np.array(aux["atom_positions"][:, 1])
        native_ca = runner.wt_batch["all_atom_positions"][:, 1]
        tm_i = tmscore(native_ca, input_ca, args.tm)
        tm_o = tmscore(native_ca, output_ca, args.tm)
        tm_io = tmscore(input_ca, output_ca, args.tm)
        composite = ptm * plddt * tm_io if tm_io is not None and not np.isnan(ptm) else np.nan

        row = {
            "target": args.target,
            "id": pdb_path.name,
            "model_name": model_name,
            "model_mode": args.model_mode,
            "version": version,
            "model_num": args.model_num,
            "recycle": args.recycle,
            "iterations": args.iterations,
            "chain": args.chain or "",
            "rm_seq": bool(args.mask_sequence),
            "rm_sc": bool(args.mask_sidechains),
            "rm_ic": bool(args.mask_interchain),
            "plddt": round(plddt, 4) if not np.isnan(plddt) else "N/A",
            "ptm": round(ptm, 4) if not np.isnan(ptm) else "N/A",
            "i_ptm": round(iptm, 4) if not np.isnan(iptm) else "N/A",
            "tm_i": round(tm_i, 4) if tm_i is not None else "N/A",
            "tm_o": round(tm_o, 4) if tm_o is not None else "N/A",
            "tm_io": round(tm_io, 4) if tm_io is not None else "N/A",
            "composite": round(float(composite), 4) if not np.isnan(composite) else "N/A",
            "pae": round(float(31.0 * log.get("pae", np.nan)), 4) if "pae" in log else "N/A",
        }
        row.update(actif_metrics)
        rows.append(row)

        if args.keep_raw:
            maybe_save_raw_npz(out_dir, pdb_id, aux, asym_id)
        save_pae_json(pae_json_dir, pdb_id, aux, gzip_enabled=args.pae_json_gzip)

        # Release large arrays between structures.
        del runner, aux, outputs

    csv_path = out_dir / f"{args.target}_af2rank_actifptm.csv"
    write_csv(rows, csv_path)

    metadata = {
        "target": args.target,
        "decoy_dir": args.decoy_dir,
        "chain": args.chain,
        "model_name": model_name,
        "params": args.params,
        "colabdesign": args.colabdesign,
        "num_pdbs": len(pdb_files),
        "csv": str(csv_path),
        "raw_npz_saved": bool(args.keep_raw),
        "iterations": args.iterations,
        "pae_json_dir": str(pae_json_dir),
        "pae_json_gzip": bool(args.pae_json_gzip),
        "scored_pdbs_saved": True,
        "actifptm_variant": ACTIFPTM_VARIANT,
    }
    with open(out_dir / f"{args.target}_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
