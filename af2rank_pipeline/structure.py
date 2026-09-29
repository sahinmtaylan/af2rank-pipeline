from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, TextIO

from .exceptions import DependencyError, PipelineError

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

ONE_TO_THREE = {value: key for key, value in THREE_TO_ONE.items() if key != "MSE"}
BACKBONE_ORDER = {"N": 0, "CA": 1, "C": 2, "O": 3}


@dataclass
class AtomRecord:
    name: str
    altloc: str
    x: float
    y: float
    z: float
    occupancy: float
    bfactor: float
    element: str


@dataclass
class ResidueRecord:
    chain_id: str
    resseq: int
    icode: str
    resname: str
    one_letter: str
    atoms: list[AtomRecord] = field(default_factory=list)

    @property
    def residue_key(self) -> tuple[str, int, str, str]:
        return (self.chain_id, self.resseq, self.icode, self.resname)


@dataclass
class ChainRecord:
    chain_id: str
    residues: list[ResidueRecord] = field(default_factory=list)

    @property
    def sequence(self) -> str:
        return "".join(residue.one_letter for residue in self.residues)


@dataclass
class StructureRecord:
    chains: list[ChainRecord]

    def chain_by_id(self, chain_id: str) -> ChainRecord:
        for chain in self.chains:
            if chain.chain_id == chain_id:
                return chain
        raise KeyError(chain_id)


def _safe_float(value: str, default: float) -> float:
    try:
        return float(value)
    except ValueError:
        return default


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except ValueError:
        return default


def _atom_sort_key(atom: AtomRecord) -> tuple[int, str]:
    return (BACKBONE_ORDER.get(atom.name.strip(), 10), atom.name.strip())


def _open_structure_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open(errors="replace")


def parse_pdb(path: str | Path, model_number: int | None = None) -> StructureRecord:
    """Parse protein ATOM/HETATM records from a PDB file.

    model_number is the literal MODEL number when present. If None, the first model
    is read. Files without MODEL records are treated as one model.
    """
    path = Path(path)
    selected_lines: list[str] = []
    in_selected_model = model_number is None
    saw_model = False

    with _open_structure_text(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            record = line[:6]
            if record == "MODEL ":
                saw_model = True
                current_model = _safe_int(line[6:14].strip(), len(selected_lines))
                in_selected_model = model_number is None or current_model == model_number
                continue
            if record == "ENDMDL":
                if in_selected_model and saw_model:
                    break
                in_selected_model = model_number is None
                continue
            if not saw_model:
                in_selected_model = True
            if in_selected_model and record in {"ATOM  ", "HETATM"}:
                selected_lines.append(line)

    return parse_pdb_lines(selected_lines)


def iter_pdb_model_lines(path: str | Path) -> Iterable[tuple[int | None, list[str]]]:
    """Yield ATOM/HETATM lines for each MODEL block in a PDB file.

    CAPRI scoring files store hundreds or thousands of models in one PDB. This
    iterator lets callers process those ensembles in a single pass instead of
    rereading the full file once per model.
    """
    path = Path(path)
    current_model: int | None = None
    current_lines: list[str] = []
    saw_model = False

    with _open_structure_text(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            record = line[:6]
            if record == "MODEL ":
                saw_model = True
                current_model = _safe_int(line[6:14].strip(), 0)
                current_lines = []
                continue
            if record == "ENDMDL":
                if saw_model and current_model is not None:
                    yield current_model, current_lines
                current_model = None
                current_lines = []
                continue
            if record in {"ATOM  ", "HETATM"}:
                if saw_model:
                    if current_model is not None:
                        current_lines.append(line)
                else:
                    current_lines.append(line)

    if not saw_model and current_lines:
        yield None, current_lines


def parse_pdb_lines(lines: Iterable[str]) -> StructureRecord:
    chains: dict[str, ChainRecord] = {}
    residue_lookup: dict[tuple[str, int, str, str], ResidueRecord] = {}
    atom_altloc_score = {" ": 3, "A": 2, "1": 1}
    atom_lookup: dict[tuple[str, int, str, str, str], tuple[int, AtomRecord]] = {}

    for line in lines:
        if len(line) < 54 or line[:6] not in {"ATOM  ", "HETATM"}:
            continue
        resname = line[17:20].strip().upper()
        if resname not in THREE_TO_ONE:
            continue
        if line[:6] == "HETATM" and resname != "MSE":
            continue

        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        atom_name = line[12:16].strip()
        if element == "H" or atom_name.startswith("H"):
            continue

        chain_id = line[21].strip() or "_"
        resseq = _safe_int(line[22:26].strip(), 0)
        icode = line[26].strip()
        normalized_resname = "MET" if resname == "MSE" else resname
        key = (chain_id, resseq, icode, normalized_resname)

        if chain_id not in chains:
            chains[chain_id] = ChainRecord(chain_id=chain_id)
        if key not in residue_lookup:
            residue = ResidueRecord(
                chain_id=chain_id,
                resseq=resseq,
                icode=icode,
                resname=normalized_resname,
                one_letter=THREE_TO_ONE[resname],
            )
            chains[chain_id].residues.append(residue)
            residue_lookup[key] = residue

        altloc = line[16] if len(line) > 16 else " "
        atom = AtomRecord(
            name=atom_name,
            altloc=altloc,
            x=_safe_float(line[30:38].strip(), 0.0),
            y=_safe_float(line[38:46].strip(), 0.0),
            z=_safe_float(line[46:54].strip(), 0.0),
            occupancy=_safe_float(line[54:60].strip(), 1.0) if len(line) >= 60 else 1.0,
            bfactor=_safe_float(line[60:66].strip(), 0.0) if len(line) >= 66 else 0.0,
            element=element or "".join(ch for ch in atom_name if ch.isalpha())[:1].upper(),
        )

        atom_key = key + (atom.name,)
        score = atom_altloc_score.get(atom.altloc, 0)
        previous = atom_lookup.get(atom_key)
        if previous is None or score > previous[0] or (
            score == previous[0] and atom.occupancy > previous[1].occupancy
        ):
            atom_lookup[atom_key] = (score, atom)

    for chain in chains.values():
        for residue in chain.residues:
            residue.atoms = [
                atom
                for atom_key, (_, atom) in atom_lookup.items()
                if atom_key[:4] == residue.residue_key
            ]
            residue.atoms.sort(key=_atom_sort_key)

    return StructureRecord(chains=list(chains.values()))


def parse_mmcif(path: str | Path, model_number: int | None = None) -> StructureRecord:
    try:
        from Bio.PDB import MMCIFParser
        from Bio.SeqUtils import seq1
    except ImportError as exc:
        raise DependencyError("mmCIF parsing requires biopython. Install the package dependencies.") from exc

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("-", str(path))
    model_idx = 0 if model_number is None else max(0, model_number - 1)
    try:
        model = list(structure)[model_idx]
    except IndexError as exc:
        raise PipelineError(f"Model {model_number} not found in {path}") from exc

    chains: list[ChainRecord] = []
    for bio_chain in model:
        chain = ChainRecord(chain_id=(bio_chain.id.strip() or "_"))
        for bio_residue in bio_chain:
            hetflag, resseq, icode = bio_residue.id
            resname = bio_residue.resname.strip().upper()
            if hetflag.strip() and resname != "MSE":
                continue
            if resname not in THREE_TO_ONE:
                continue
            normalized_resname = "MET" if resname == "MSE" else resname
            residue = ResidueRecord(
                chain_id=chain.chain_id,
                resseq=int(resseq),
                icode=icode.strip(),
                resname=normalized_resname,
                one_letter=THREE_TO_ONE[resname],
            )
            seen_atoms: set[str] = set()
            for bio_atom in bio_residue.get_unpacked_list():
                atom_name = bio_atom.get_name().strip()
                element = (bio_atom.element or "").strip().upper()
                if element == "H" or atom_name.startswith("H") or atom_name in seen_atoms:
                    continue
                seen_atoms.add(atom_name)
                coord = bio_atom.coord
                residue.atoms.append(
                    AtomRecord(
                        name=atom_name,
                        altloc=(bio_atom.altloc or " "),
                        x=float(coord[0]),
                        y=float(coord[1]),
                        z=float(coord[2]),
                        occupancy=float(bio_atom.occupancy or 1.0),
                        bfactor=float(bio_atom.bfactor or 0.0),
                        element=element or atom_name[:1],
                    )
                )
            residue.atoms.sort(key=_atom_sort_key)
            if residue.atoms:
                chain.residues.append(residue)
        if chain.residues:
            chains.append(chain)
    return StructureRecord(chains=chains)


def load_structure(path: str | Path, model_number: int | None = None) -> StructureRecord:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".gz":
        suffix = Path(path.stem).suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        if path.suffix.lower() == ".gz":
            raise PipelineError(f"Compressed mmCIF input is not supported: {path}")
        return parse_mmcif(path, model_number=model_number)
    return parse_pdb(path, model_number=model_number)


@dataclass
class CleanResidue:
    target_chain_id: str
    flat_index: int
    local_index: int
    resname: str
    source_chain_id: str
    source_resseq: int
    source_icode: str
    atoms: list[AtomRecord]


def format_pdb_atom_line(
    serial: int,
    atom: AtomRecord,
    resname: str,
    chain_id: str,
    resseq: int,
) -> str:
    atom_name = atom.name.strip()
    if len(atom_name) < 4 and atom_name[:1].isalpha():
        atom_field = f" {atom_name:<3}"
    else:
        atom_field = f"{atom_name:>4}"
    return (
        f"ATOM  {serial:5d} {atom_field} {resname:>3} {chain_id:1s}"
        f"{resseq:4d}    "
        f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
        f"{atom.occupancy:6.2f}{atom.bfactor:6.2f}          "
        f"{atom.element[:2]:>2s}\n"
    )


def write_clean_pdb(clean_residues: list[CleanResidue], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    serial = 1
    previous_chain = None
    previous_residue: CleanResidue | None = None

    for residue in clean_residues:
        if previous_chain is not None and residue.target_chain_id != previous_chain:
            assert previous_residue is not None
            lines.append(
                f"TER   {serial:5d}      {previous_residue.resname:>3} {previous_chain:1s}"
                f"{previous_residue.local_index:4d}\n"
            )
            serial += 1
        for atom in residue.atoms:
            lines.append(
                format_pdb_atom_line(
                    serial=serial,
                    atom=atom,
                    resname=residue.resname,
                    chain_id=residue.target_chain_id,
                    resseq=residue.local_index,
                )
            )
            serial += 1
        previous_chain = residue.target_chain_id
        previous_residue = residue

    if previous_residue is not None and previous_chain is not None:
        lines.append(
            f"TER   {serial:5d}      {previous_residue.resname:>3} {previous_chain:1s}"
            f"{previous_residue.local_index:4d}\n"
        )
    lines.append("END\n")
    path.write_text("".join(lines))
