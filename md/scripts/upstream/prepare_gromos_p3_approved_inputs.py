#!/usr/bin/env python3
"""Build the immutable aDMan production input and its P3-only approval manifest.

This release intentionally knows about one target only.  It cannot approve the
other 18 alignment-364 GROMOS gaps by name, topology presence, or a successful
``grompp``.  Every approval is regenerated from the current P3 status and all
of its hash-bound evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from gromos_curated_gate import (
    APPROVED_MANIFEST_SCHEMA,
    DEFAULT_APPROVED_MANIFEST,
    MD_ROOT,
    canonical_json_bytes,
    sha256_directory_tree,
    sha256_file,
    validate_p3_status,
)


ROOT = MD_ROOT.parent
P1 = MD_ROOT / "md_alignment364_v1" / "gromos" / "p1_adman"
P3_STATUS = P1 / "P3_STATUS.json"
NATIVE_TOP = P1 / "native_reference" / "aDMan_56A6_CARBO_R.top"
NATIVE_AUDIT = P1 / "native_reference" / "aDMan_native_audit.json"
DIAGNOSTIC_TOP = P1 / "p2_diagnostic" / "aDMan_56A6_CARBO_R_DIAGNOSTIC.top"
DIAGNOSTIC_GRO = P1 / "p2_diagnostic" / "conformers" / "aDMan_conf00.gro"
DIAGNOSTIC_MAPPING = P1 / "p2_diagnostic" / "sdf_to_native17_mapping.json"
P3_EQUIVALENCE = P1 / "p3_diagnostic" / "equivalence_results.json"
P3_SMOKE = P1 / "p3_diagnostic" / "smoke_results.json"
CURATED_ROOT = P1 / "curated_production_input_v1"
CURATED_TOP = CURATED_ROOT / "aDMan_GROMOS56A6CARBOR.top"
CURATED_GRO = CURATED_ROOT / "aDMan_GROMOS56A6CARBOR_processed.gro"
CURATED_MAP = CURATED_ROOT / "sdf_to_gromos_frame_heavy_index_map.json"
CURATED_METADATA = CURATED_ROOT / "gromos_curated_input_metadata.json"
CURATED_FF = CURATED_ROOT / "gromos56a6carbo_curated.ff"
CURATED_SPC = CURATED_FF / "spc.itp"
PACKAGE_MANIFEST = CURATED_ROOT / "curated_input_manifest.json"
SOURCE_SDF = ROOT / "data" / "dataset" / "casper_mono.sdf"
GENERATOR = Path(__file__).resolve()

EXPECTED_P3_STATUS_SHA256 = "2ed592eacea28921eb509490fb39272e4f541f94c6598ad5b6144361e4c36c68"
EXPECTED_DIAGNOSTIC_TOP_SHA256 = "2b30339bf09c28d128fc1d419a450cd696190237b78f852c83a338f69f79d6aa"
EXPECTED_NATIVE_TOP_SHA256 = "9fa826e5212f624780e152fd62c92b8e4c2f974b47386ba5c5d08415a2aede30"
EXPECTED_START_GRO_SHA256 = "5626f2467435ce1b7699e84a151cc24c28ee9495ca0c44690e6b3498d103a755"
EXPECTED_MAPPING_SHA256 = "c8f23404e9955993ee489fcd7e8c295b7a08e7d59e2a4666fa14f0ce0e0cf1d2"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} changed: expected SHA256 {expected}, observed {actual}: {path}"
        )


def relative_to_md(path: Path) -> str:
    return path.resolve().relative_to(MD_ROOT.resolve()).as_posix()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_frozen(path: Path, payload: bytes) -> None:
    """Create a versioned curated artifact once; never rewrite changed bytes."""

    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"frozen output is not a regular file: {path}")
        if path.read_bytes() != payload:
            raise RuntimeError(
                f"immutable curated v1 artifact differs: {path}; create a new input version"
            )
        return
    _atomic_write(path, payload)


def write_reproducible_manifest(path: Path, value: object) -> None:
    payload = canonical_json_bytes(value)
    if path.exists() and path.read_bytes() == payload:
        return
    _atomic_write(path, payload)


def native_tables() -> tuple[dict[int, str], dict[tuple[int, int], dict[str, str]], list[list[str]]]:
    sys.path.insert(0, str(P1))
    from import_native_topology import parse_atom_types, parse_lj_table, parse_sections, rows

    sections = parse_sections(NATIVE_TOP)
    return (
        parse_atom_types(sections),
        parse_lj_table(sections),
        rows(sections, "SOLVENTATOM"),
    )


def render_spc() -> str:
    atom_types, _lj, solvent_rows = native_tables()
    if atom_types[5] != "OW" or atom_types[21] != "H":
        raise RuntimeError("native 56A6@CARBO-R solvent IAC identity changed")
    if solvent_rows != [
        ["3"],
        ["1", "OW", "5", "15.99940", "-0.82000"],
        ["2", "HW1", "21", "1.00800", "0.41000"],
        ["3", "HW2", "21", "1.00800", "0.41000"],
    ]:
        raise RuntimeError("native 56A6@CARBO-R SOLVENTATOM block changed")
    return """; Exact SPC solvent from native GROMOS 56A6@CARBO-R topology.
[ moleculetype ]
; molname nrexcl
SOL 2

[ atoms ]
; nr type          resnr residue atom cgnr charge mass
1 G56_IAC005       1     SOL     OW   1   -0.82000 15.99940
2 G56_IAC021       1     SOL     HW1  1    0.41000  1.00800
3 G56_IAC021       1     SOL     HW2  1    0.41000  1.00800

[ settles ]
; OW funct dOH[nm] dHH[nm]
1 1 0.1000000 0.1632990

[ exclusions ]
1 2 3
2 1 3
3 1 2
"""


def render_production_topology() -> str:
    text = DIAGNOSTIC_TOP.read_text(encoding="utf-8")
    required_markers = (
        "; DIAGNOSTIC ONLY / NOT PRODUCTION / NO MD AUTHORIZATION",
        "ADMAN_DIAGNOSTIC 0",
        "aDMan 56A6@CARBO-R diagnostic conversion; NOT PRODUCTION",
        "ADMAN_DIAGNOSTIC 1",
    )
    if any(text.count(marker) != 1 for marker in required_markers):
        raise RuntimeError("validated diagnostic topology markers changed")

    atom_types, lj_table, _solvent = native_tables()
    used_iacs = (3, 5, 15, 21, 54, 56)
    if [atom_types[index] for index in used_iacs] != ["OA", "OW", "CH2", "H", "Or", "CH1r"]:
        raise RuntimeError("native IAC names needed by curated solvent topology changed")
    ow_self = lj_table[(5, 5)]
    ow_atomtype = (
        f"G56_IAC005     8    15.999400 0.0 A "
        f"{float(ow_self['C6']):.12e} {float(ow_self['C12']):.12e}"
    )
    ow_pairs = []
    for other in used_iacs:
        row = lj_table[tuple(sorted((5, other)))]
        ow_pairs.append(
            f"G56_IAC005   G56_IAC{other:03d}   1 "
            f"{float(row['C6']):.12e} {float(row['C12']):.12e}"
        )

    text = text.replace(
        "; DIAGNOSTIC ONLY / NOT PRODUCTION / NO MD AUTHORIZATION",
        "; IMMUTABLE CURATED PRODUCTION INPUT / P3-GATED",
    ).replace(
        "; Every bonded/nonbonded term is explicit for cross-engine auditing.",
        "; Solute numerical terms are the P3-validated conversion; native SPC terms are appended.",
    )
    atomtype_header = "; name  at.num  mass  charge  ptype  C6  C12"
    text = text.replace(atomtype_header, atomtype_header + "\n" + ow_atomtype, 1)
    nonbond_header = "; i j funct C6 C12"
    text = text.replace(nonbond_header, nonbond_header + "\n" + "\n".join(ow_pairs), 1)
    text = text.replace("ADMAN_DIAGNOSTIC 0", "ADMAN_56A6CARBOR 0")
    text = text.replace(
        "[ system ]",
        '#include "gromos56a6carbo_curated.ff/spc.itp"\n\n[ system ]',
        1,
    )
    text = text.replace(
        "aDMan 56A6@CARBO-R diagnostic conversion; NOT PRODUCTION",
        "aDMan GROMOS 56A6@CARBO-R P3-validated curated production input",
    )
    text = text.replace("ADMAN_DIAGNOSTIC 1", "ADMAN_56A6CARBOR 1")
    if "DIAGNOSTIC" in text or "NOT PRODUCTION" in text:
        raise RuntimeError("diagnostic label leaked into curated topology")
    return text


def render_processed_gro() -> str:
    lines = DIAGNOSTIC_GRO.read_text(encoding="utf-8").splitlines()
    if len(lines) != 20 or lines[1].strip() != "17":
        raise RuntimeError("validated aDMan start GRO changed shape")
    output = ["aDMan GROMOS 56A6@CARBO-R P3-gated curated start", "17"]
    for raw in lines[2:19]:
        fields = raw.split()
        residue = re.fullmatch(r"(\d+)([A-Za-z0-9]+)", fields[0])
        if residue is None or len(fields) != 6:
            raise RuntimeError(f"cannot parse validated start GRO atom line: {raw}")
        residue_number = int(residue.group(1))
        residue_name = residue.group(2)
        atom_name = fields[1]
        atom_number = int(fields[2])
        x, y, z = map(float, fields[3:])
        output.append(
            f"{residue_number:5d}{residue_name:<5s}{atom_name:>5s}{atom_number:5d}"
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    box = [float(value) for value in lines[-1].split()]
    if len(box) != 3:
        raise RuntimeError("validated start GRO box changed shape")
    output.append("".join(f"{value:10.5f}" for value in box))
    return "\n".join(output) + "\n"


def render_heavy_atom_map() -> list[int]:
    mapping = load_json(DIAGNOSTIC_MAPPING)
    rows = mapping.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("P2 SDF/native mapping rows are missing")
    heavy_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("source_sdf_heavy_atom_index_zero_based"), int)
    ]
    heavy_rows.sort(key=lambda row: int(row["native_atom_number"]))
    frame_index_by_source = {
        int(row["source_sdf_heavy_atom_index_zero_based"]): frame_index
        for frame_index, row in enumerate(heavy_rows)
    }
    if set(frame_index_by_source) != set(range(12)):
        raise RuntimeError("P2 mapping does not cover all 12 source heavy atoms exactly")
    result = [frame_index_by_source[index] for index in range(12)]
    if result != [7, 6, 8, 9, 10, 11, 2, 3, 4, 5, 1, 0]:
        raise RuntimeError("derived SDF-to-GROMOS heavy-atom map changed")
    return result


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": relative_to_md(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def source_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_metadata() -> dict[str, object]:
    equivalence = load_json(P3_EQUIVALENCE)
    source_provenance = load_json(P1 / "P3_SOURCE_PROVENANCE.json")
    return {
        "schema": "gromos_curated_production_input/v1",
        "molecule_name": "aDMan",
        "clean_name": "aDMan",
        "source_class": "mono",
        "n_heavy_atoms": 12,
        "native_site_count": 17,
        "expected_solute_charge_e": 0,
        "observed_topology_charge_e": 0.0,
        "production_ready": True,
        "production_md_permitted": True,
        "five_ns_started": False,
        "parameterization_tier": "official_standard",
        "topology_provenance": "official_forcefield_template",
        "curated_template_ids": ["HO4C", "MA4P", "C1OH"],
        "curated_source_sha256": sha256_file(NATIVE_TOP),
        "experimental_gromos": False,
        "graph_derived_parameters": False,
        "immutable_input_version": 1,
        "approval_basis": "P3_STATUS_gates",
        "p3_status_path": str(P3_STATUS.resolve()),
        "p3_status_sha256": sha256_file(P3_STATUS),
        "strict_validation": {
            "passed": True,
            "checks": {
                "source_identity": True,
                "atom_identity": True,
                "stereochemistry": True,
                "formal_charge": True,
                "topology_preprocess": True,
                "energy_minimization": True,
                "short_nvt": True,
            },
            "p3_equivalence_status": equivalence["status"],
            "validated_conformer_count": len(equivalence["conformers"]),
            "identity_mapping": equivalence["atom_mapping"]["status"],
            "grompp_policy": "P3 EM and NVT both passed with -maxwarn 0",
        },
        "official_source": {
            "forcefield": "GROMOS 56A6@CARBO-R",
            "native_sequence": ["HO4C", "MA4P", "C1OH"],
            "native_topology": source_record(NATIVE_TOP),
            "native_audit": source_record(NATIVE_AUDIT),
            "mdpp_release_tag": source_provenance["official_distribution"]["tag"],
            "mdpp_tag_commit": source_provenance["official_distribution"]["tag_commit"],
            "mdpp_source_archive_sha256": source_provenance["source_archive"]["sha256"],
            "mdpp_binary_sha256": source_provenance["isolated_build"]["binary"]["sha256"],
            "license": source_provenance["license"],
        },
        "conversion_provenance": {
            "validated_diagnostic_topology": source_record(DIAGNOSTIC_TOP),
            "p3_equivalence": source_record(P3_EQUIVALENCE),
            "p3_smoke": source_record(P3_SMOKE),
            "solute_transform": (
                "Only labels/moleculetype identity and preprocessor provenance markers changed; "
                "all P3-validated solute numerical records remain byte-for-byte text copies."
            ),
        },
        "water_model": {
            "name": "SPC",
            "source": "native topology SOLVENTATOM/SOLVENTCONSTR and LJPARAMETERS",
            "oxygen_iac": 5,
            "hydrogen_iac": 21,
            "solute_water_normal_lj_pairs_explicit": True,
            "native_oh_constraint_nm": 0.1,
            "native_hh_constraint_nm": 0.163299,
        },
        "source_structure": {
            "sdf": source_record(SOURCE_SDF),
            "record_index_zero_based": 94,
            "mapping": source_record(DIAGNOSTIC_MAPPING),
            "start_coordinate": source_record(DIAGNOSTIC_GRO),
        },
        "generator": {
            "path": str(GENERATOR),
            "sha256": sha256_file(GENERATOR),
        },
        "formal_protocol_scope": {
            "solvent": "explicit SPC water",
            "production_duration_ps": 5000.0,
            "candidate_count": 200,
            "selected_count": 100,
            "current_stage": "approved_for_formal_setup; 5 ns not started by this builder",
        },
    }


def build_package() -> dict[str, object]:
    require_hash(P3_STATUS, EXPECTED_P3_STATUS_SHA256, "P3 status")
    validate_p3_status(P3_STATUS)
    require_hash(DIAGNOSTIC_TOP, EXPECTED_DIAGNOSTIC_TOP_SHA256, "P3 topology")
    require_hash(NATIVE_TOP, EXPECTED_NATIVE_TOP_SHA256, "native topology")
    require_hash(DIAGNOSTIC_GRO, EXPECTED_START_GRO_SHA256, "P3 start coordinate")
    require_hash(DIAGNOSTIC_MAPPING, EXPECTED_MAPPING_SHA256, "P3 atom mapping")

    write_frozen(CURATED_TOP, render_production_topology().encode("utf-8"))
    write_frozen(CURATED_GRO, render_processed_gro().encode("utf-8"))
    write_frozen(
        CURATED_MAP,
        (json.dumps(render_heavy_atom_map(), separators=(",", ":")) + "\n").encode("utf-8"),
    )
    write_frozen(CURATED_SPC, render_spc().encode("utf-8"))
    write_frozen(
        CURATED_METADATA,
        (json.dumps(build_metadata(), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )

    package_body: dict[str, object] = {
        "schema": "gromos_curated_input_package/v1",
        "clean_name": "aDMan",
        "molecule_name": "aDMan",
        "forcefield": "GROMOS 56A6@CARBO-R",
        "immutable_input_version": 1,
        "approval_basis": "P3_STATUS_gates",
        "p3_status": source_record(P3_STATUS),
        "files": {
            "processed_gro": file_record(CURATED_GRO),
            "topology": file_record(CURATED_TOP),
            "atom_map": file_record(CURATED_MAP),
            "metadata": file_record(CURATED_METADATA),
            "spc_itp": file_record(CURATED_SPC),
        },
        "forcefield_tree": {
            "path": relative_to_md(CURATED_FF),
            "tree": sha256_directory_tree(CURATED_FF),
        },
    }
    package = {
        **package_body,
        "package_id": hashlib.sha256(canonical_json_bytes(package_body)).hexdigest(),
    }
    write_frozen(PACKAGE_MANIFEST, canonical_json_bytes(package))

    expected_files = {
        CURATED_TOP,
        CURATED_GRO,
        CURATED_MAP,
        CURATED_METADATA,
        CURATED_SPC,
        PACKAGE_MANIFEST,
    }
    observed_files = {path for path in CURATED_ROOT.rglob("*") if path.is_file()}
    if observed_files != expected_files:
        raise RuntimeError(
            "immutable curated input directory contains unexpected files: "
            + ", ".join(str(path.relative_to(CURATED_ROOT)) for path in sorted(observed_files - expected_files))
        )
    return package


def build_approved_manifest(package: dict[str, object]) -> dict[str, object]:
    files = package["files"]
    assert isinstance(files, dict)
    forcefield = package["forcefield_tree"]
    assert isinstance(forcefield, dict)
    body: dict[str, object] = {
        "schema": APPROVED_MANIFEST_SCHEMA,
        "policy": (
            "Entries are generated only from a complete hash-valid P3_STATUS gate; "
            "topology presence, grompp, EM, or a manually set boolean cannot approve a target."
        ),
        "generator": {
            "path": relative_to_md(GENERATOR),
            "sha256": sha256_file(GENERATOR),
        },
        "approved_count": 1,
        "explicitly_not_approved_count": 18,
        "entries": [
            {
                "clean_name": "aDMan",
                "molecule_name": "aDMan",
                "source_class": "mono",
                "n_heavy_atoms": 12,
                "priority": 83,
                "repair_forcefield": "gromos",
                "approval_basis": "P3_STATUS_gates",
                "p3_status": {
                    "path": relative_to_md(P3_STATUS),
                    "sha256": sha256_file(P3_STATUS),
                },
                "package_manifest": {
                    "path": relative_to_md(PACKAGE_MANIFEST),
                    "sha256": sha256_file(PACKAGE_MANIFEST),
                    "package_id": package["package_id"],
                },
                "inputs": {
                    "processed_gro": files["processed_gro"],
                    "topology": files["topology"],
                    "metadata": files["metadata"],
                    "atom_map": files["atom_map"],
                    "forcefield": forcefield,
                },
            }
        ],
    }
    manifest = {
        **body,
        "manifest_id": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }
    write_reproducible_manifest(DEFAULT_APPROVED_MANIFEST, manifest)
    return manifest


def main() -> None:
    package = build_package()
    manifest = build_approved_manifest(package)
    print(
        json.dumps(
            {
                "curated_root": str(CURATED_ROOT),
                "package_id": package["package_id"],
                "approved_manifest": str(DEFAULT_APPROVED_MANIFEST),
                "manifest_id": manifest["manifest_id"],
                "approved_targets": ["aDMan"],
                "other_gromos_targets_approved": 0,
                "five_ns_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
