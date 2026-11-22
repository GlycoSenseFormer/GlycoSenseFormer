#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import re
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem


def clean_name(name: str) -> str:
    return name.replace("->", "").replace("(", "").replace(")", "")


def infer_element(atom_name: str, element: str = "") -> str:
    element = element.strip()
    if element:
        return element.upper()
    name = atom_name.strip()
    if name and name[0].isdigit():
        name = name[1:]
    if len(name) >= 2 and name[:2].upper() in {"CL", "BR"}:
        return name[:2].upper()
    return name[:1].upper()


def is_heavy(atom_name: str, element: str = "") -> bool:
    return infer_element(atom_name, element) != "H"


def read_pdb_heavy_coords(path: Path, scale: float) -> np.ndarray:
    coords: list[list[float]] = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16]
            element = line[76:78] if len(line) >= 78 else ""
            if not is_heavy(atom_name, element):
                continue
            coords.append(
                [
                    float(line[30:38]) * scale,
                    float(line[38:46]) * scale,
                    float(line[46:54]) * scale,
                ]
            )
    return np.asarray(coords, dtype=np.float32)


def read_gro_heavy_coords(path: Path, scale: float) -> np.ndarray:
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path} is too short to be a .gro file")
    try:
        natoms = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(f"Cannot parse atom count from {path}") from exc
    atom_lines = lines[2 : 2 + natoms]
    coords: list[list[float]] = []
    for line in atom_lines:
        atom_name = line[10:15]
        if not is_heavy(atom_name):
            continue
        coords.append(
            [
                float(line[20:28]) * scale,
                float(line[28:36]) * scale,
                float(line[36:44]) * scale,
            ]
        )
    return np.asarray(coords, dtype=np.float32)


def unit_scale(path: Path, input_unit: str) -> float:
    if input_unit == "angstrom":
        return 1.0
    if input_unit == "nm":
        return 10.0
    suffix = path.suffix.lower()
    if suffix == ".gro":
        return 10.0
    return 1.0


def read_heavy_coords(path: Path, input_unit: str) -> np.ndarray:
    scale = unit_scale(path, input_unit)
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return read_pdb_heavy_coords(path, scale)
    if suffix == ".gro":
        return read_gro_heavy_coords(path, scale)
    raise ValueError(f"Unsupported frame format: {path}. Use .pdb or .gro")


def find_reference_mol(reference_sdf: Path, molecule_name: str | None):
    supplier = Chem.SDMolSupplier(str(reference_sdf), removeHs=False)
    first = None
    for mol in supplier:
        if mol is None:
            continue
        if first is None:
            first = mol
        if molecule_name is None:
            continue
        if clean_name(mol.GetProp("_Name")) == clean_name(molecule_name):
            return mol
    if molecule_name is None and first is not None:
        return first
    raise ValueError(f"Could not find molecule {molecule_name!r} in {reference_sdf}")


def heavy_atom_count(mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def load_index_map(path: Path | None, n_heavy: int) -> list[int] | None:
    if path is None:
        return None
    mapping = json.loads(path.read_text())
    if not isinstance(mapping, list) or not all(isinstance(i, int) for i in mapping):
        raise ValueError("atom_index_map must be a JSON list of integer frame indices")
    if len(mapping) != n_heavy:
        raise ValueError(
            f"atom_index_map length {len(mapping)} does not match SDF heavy atom count {n_heavy}"
        )
    if len(set(mapping)) != len(mapping):
        raise ValueError("atom_index_map contains duplicate frame indices")
    return mapping


def select_frames(paths: list[Path], stride: int, max_frames: int | None) -> list[Path]:
    selected = paths[::stride]
    if max_frames is not None:
        selected = selected[:max_frames]
    return selected


def natural_sort_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_set_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def build_metadata(
    logical_output: Path,
    *,
    reference_sdf: Path,
    molecule_name: str | None,
    frames: list[Path],
    n_heavy: int,
    input_unit: str,
    atom_index_map: Path | None,
    pickle_sha256: str,
    epoch_id: str | None,
    epoch_manifest_sha256: str | None,
    artifact_role: str | None,
) -> dict[str, object]:
    return {
        "output": str(logical_output),
        "reference_sdf": str(reference_sdf),
        "reference_sdf_sha256": sha256_file(reference_sdf),
        "molecule_name": molecule_name,
        "n_frames": len(frames),
        "n_heavy_atoms": n_heavy,
        "coordinate_unit": "angstrom",
        "input_unit": input_unit,
        "mapping_used": atom_index_map is not None,
        "atom_index_map": str(atom_index_map) if atom_index_map is not None else None,
        "atom_index_map_sha256": sha256_file(atom_index_map) if atom_index_map is not None else None,
        "frame_set_sha256": frame_set_sha256(frames),
        "pickle_sha256": pickle_sha256,
        "epoch_id": epoch_id,
        "epoch_manifest_sha256": epoch_manifest_sha256,
        "artifact_role": artifact_role,
        "frames": [str(path) for path in frames],
    }


def write_pickle_and_metadata_atomic(
    output: Path,
    conformers: list[np.ndarray],
    *,
    logical_output: Path,
    reference_sdf: Path,
    molecule_name: str | None,
    frames: list[Path],
    n_heavy: int,
    input_unit: str,
    atom_index_map: Path | None,
    epoch_id: str | None,
    epoch_manifest_sha256: str | None,
    artifact_role: str | None,
) -> None:
    """Write metadata first and the pickle last using same-directory renames."""

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output = output.with_suffix(output.suffix + ".meta.json")
    pickle_descriptor, pickle_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    pickle_temporary = Path(pickle_name)
    metadata_temporary: Path | None = None
    try:
        with os.fdopen(pickle_descriptor, "wb") as handle:
            pickle.dump(conformers, handle)
            handle.flush()
            os.fsync(handle.fileno())
        pickle_hash = sha256_file(pickle_temporary)
        metadata = build_metadata(
            logical_output,
            reference_sdf=reference_sdf,
            molecule_name=molecule_name,
            frames=frames,
            n_heavy=n_heavy,
            input_unit=input_unit,
            atom_index_map=atom_index_map,
            pickle_sha256=pickle_hash,
            epoch_id=epoch_id,
            epoch_manifest_sha256=epoch_manifest_sha256,
            artifact_role=artifact_role,
        )
        metadata_descriptor, metadata_name = tempfile.mkstemp(
            prefix=f".{metadata_output.name}.", suffix=".tmp", dir=output.parent
        )
        metadata_temporary = Path(metadata_name)
        with os.fdopen(metadata_descriptor, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(metadata_temporary, metadata_output)
        os.replace(pickle_temporary, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        pickle_temporary.unlink(missing_ok=True)
        if metadata_temporary is not None:
            metadata_temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert solute-only GROMACS frame PDB/GRO files to this project's conformer pickle format."
    )
    parser.add_argument("--frames_glob", required=True, help="Glob for single-frame .pdb or .gro files")
    parser.add_argument("--reference_sdf", required=True, help="Original CASPER SDF used by data/create_data.py")
    parser.add_argument("--molecule_name", default=None, help="Molecule _Name in the SDF. Defaults to first molecule.")
    parser.add_argument("--output", required=True, help="Output pickle path")
    parser.add_argument(
        "--atom_index_map",
        default=None,
        help="Optional JSON list mapping SDF heavy atom order to frame heavy atom order.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth matched frame")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to write")
    parser.add_argument(
        "--input_unit",
        choices=("auto", "angstrom", "nm"),
        default="auto",
        help="Coordinate unit in frame files. auto treats .gro as nm and .pdb as Angstrom.",
    )
    parser.add_argument("--logical_output", default="", help="Logical/canonical output recorded in metadata")
    parser.add_argument("--epoch_id", default="", help="Optional SHA256 epoch identifier")
    parser.add_argument("--epoch_manifest_sha256", default="", help="Optional output-epoch manifest SHA256")
    parser.add_argument("--artifact_role", choices=("all", "low"), default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    output = Path(args.output)
    reference_sdf = Path(args.reference_sdf)
    frame_paths = sorted((Path(path) for path in glob.glob(args.frames_glob)), key=natural_sort_key)
    frame_paths = select_frames(frame_paths, args.stride, args.max_frames)
    if not frame_paths:
        raise ValueError(f"No frames matched {args.frames_glob!r}")

    mol = find_reference_mol(reference_sdf, args.molecule_name)
    n_heavy = heavy_atom_count(mol)
    atom_index_map = Path(args.atom_index_map) if args.atom_index_map else None
    mapping = load_index_map(atom_index_map, n_heavy)

    conformers: list[np.ndarray] = []
    for frame in frame_paths:
        coords = read_heavy_coords(frame, args.input_unit)
        if coords.shape[0] < n_heavy:
            raise ValueError(
                f"{frame} has {coords.shape[0]} heavy atoms, fewer than SDF heavy atom count {n_heavy}"
            )
        if mapping is None:
            if coords.shape[0] != n_heavy:
                raise ValueError(
                    f"{frame} has {coords.shape[0]} heavy atoms but SDF has {n_heavy}. "
                    "Provide --atom_index_map if the frame contains extra heavy atoms."
                )
            ordered = coords
        else:
            if max(mapping) >= coords.shape[0]:
                raise ValueError(f"atom_index_map references atom {max(mapping)} but {frame} has {coords.shape[0]}")
            ordered = coords[np.asarray(mapping, dtype=np.int64)]
        conformers.append(np.asarray(ordered, dtype=np.float32))

    write_pickle_and_metadata_atomic(
        output,
        conformers,
        logical_output=Path(args.logical_output) if args.logical_output else output,
        reference_sdf=reference_sdf,
        molecule_name=args.molecule_name,
        frames=frame_paths,
        n_heavy=n_heavy,
        input_unit=args.input_unit,
        atom_index_map=atom_index_map,
        epoch_id=args.epoch_id or None,
        epoch_manifest_sha256=args.epoch_manifest_sha256 or None,
        artifact_role=args.artifact_role,
    )
    print(f"Wrote {len(conformers)} conformers to {output}")
    print(f"Each conformer shape: ({n_heavy}, 3), unit: Angstrom")


if __name__ == "__main__":
    main()
