#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from audit_glycam_chirality import audit_chirality
from repair_glycam_coordinates_from_mmff import repair_coordinates, sha256


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
SAMPLE_CSV = Path(os.environ.get("MD_SAMPLE_CSV", MD_ROOT / "md_small_sample_dataset_v1" / "md_small_sample_glycans_v1.csv"))
RUN_ROOT = Path(os.environ.get("MD_RUN_ROOT", MD_ROOT / "md_runs_extended24"))
TLEAP = Path("/home/cns/.local/lib/python3.13/site-packages/acpype/amber_linux/bin/tleap")
ACPYPE = Path("/home/cns/.local/bin/acpype")
AMBERHOME = TLEAP.parent.parent
GLYCAM_LEAPRC = AMBERHOME / "dat" / "leap" / "cmd" / "leaprc.GLYCAM_06j-1"
GLYCAM_PREP = AMBERHOME / "dat" / "leap" / "prep" / "GLYCAM_06j-1.prep"


GLYCAM_SEQUENCES: dict[str, list[str]] = {
    "aDGlcOMe": ["OME", "0GA"],
    "bDGlcOMe": ["OME", "0GB"],
    "aDManOMe": ["OME", "0MA"],
    "bDGalOMe": ["OME", "0LB"],
    "bDGlcNAcOMe": ["OME", "0YB"],
    "aLFucOMe": ["OME", "0fA"],
    "aDGlc12aDGlcOMe": ["OME", "2GA", "0GA"],
    "aDGlc13aDGlcOMe": ["OME", "3GA", "0GA"],
    "bDGlc14bDGlcOMe": ["OME", "4GB", "0GB"],
    "aDGlc16aDGlcOMe": ["OME", "6GA", "0GA"],
    "aDMan13bDGalOMe": ["OME", "3LB", "0MA"],
    "bDGlc(1->2)[bDGlc(1->3)]aDGlcOMe": ["OME", "ZGA", "0GB", "0GB"],
    "aLRhaOMe": ["OME", "0hA"],
    "bDXylOMe": ["OME", "0XB"],
    "aDGalOMe": ["OME", "0LA"],
    "bDGlc12bDGlcOMe": ["OME", "2GB", "0GB"],
    "bDGlc13bDGlcOMe": ["OME", "3GB", "0GB"],
    "aDGlc14bDGlcOMe": ["OME", "4GB", "0GA"],
    "bDGlcNAc14bDGlcNAc": ["ROH", "4YB", "0YB"],
    "aLFuc12aDGal": ["ROH", "2LA", "0fA"],
    "bDGal14bDGlcOMe": ["OME", "4GB", "0LB"],
    "aDMan12aDManOMe": ["OME", "2MA", "0MA"],
    "aDMan(1->2)aDMan(1->3)aDManOMe": ["OME", "3MA", "2MA", "0MA"],
    "aLFuc(1->2)bDGal(1->4)bDGlcNAcOMe": ["OME", "4YB", "2LB", "0fA"],
    "bLFucOMe": ["OME", "0fB"],
    "bLRhaOMe": ["OME", "0hB"],
    "bDManOMe": ["OME", "0MB"],
    "bLFuc12bDGalOMe": ["OME", "2LB", "0fB"],
    "aDGal14bDGlcOMe": ["OME", "4GB", "0LA"],
    "aDMan14bDGlcOMe": ["OME", "4GB", "0MA"],
    "aDGlc12bDGlcOMe": ["OME", "2GB", "0GA"],
    "bDGlc12aDGlcOMe": ["OME", "2GA", "0GB"],
    "aDGlc13bDGlcOMe": ["OME", "3GB", "0GA"],
    "bDGlc14aDGlcOMe": ["OME", "4GA", "0GB"],
    "aDMan(1->2)aDMan(1->2)aDManOMe": ["OME", "2MA", "2MA", "0MA"],
    "aDGlc(1->2)aDGlc(1->3)aDGlcOMe": ["OME", "3GA", "2GA", "0GA"],
    "aDGlc12aDManOMe": ["OME", "2MA", "0GA"],
    "aDGlc13aDManOMe": ["OME", "3MA", "0GA"],
    "aDGlc16aDGalOMe": ["OME", "6LA", "0GA"],
    "aDGlc16bDGlcOMe": ["OME", "6GB", "0GA"],
    "aDMan12aDGlcOMe": ["OME", "2GA", "0MA"],
    "aDMan12bDGlcOMe": ["OME", "2GB", "0MA"],
    "aDMan13aDManOMe": ["OME", "3MA", "0MA"],
    "aDMan13bDGlcOMe": ["OME", "3GB", "0MA"],
    "bDGlc13aDGlcOMe": ["OME", "3GA", "0GB"],
    "bDGlc13aDManOMe": ["OME", "3MA", "0GB"],
    "aDMan(1->2)aDMan(1->6)aDManOMe": ["OME", "6MA", "2MA", "0MA"],
    "bDGlc(1->2)bDGlc(1->3)aDGlcOMe": ["OME", "3GA", "2GB", "0GB"],
    "aLRha12aLRhaOMe": ["OME", "2hA", "0hA"],
    "aLRha13aLRhaOMe": ["OME", "3hA", "0hA"],
    "bLRha12aLRhaOMe": ["OME", "2hA", "0hB"],
    "aDGlc13aLRhaOMe": ["OME", "3hA", "0GA"],
    "bDGlc13aLRhaOMe": ["OME", "3hA", "0GB"],
    "aDMan12aLRhaOMe": ["OME", "2hA", "0MA"],
    "aDMan13aLRhaOMe": ["OME", "3hA", "0MA"],
    "bLFuc12aLRhaOMe": ["OME", "2hA", "0fB"],
    "bLFuc13aLRhaOMe": ["OME", "3hA", "0fB"],
    "aLRha(1->2)aLRha(1->2)aLRhaOMe": ["OME", "2hA", "2hA", "0hA"],
    "aDGlc(1->2)bDGlc(1->3)aDGlcOMe": ["OME", "3GA", "2GB", "0GA"],
    "bDGlc(1->2)aDGlc(1->3)aDGlcOMe": ["OME", "3GA", "2GA", "0GB"],
    "aDGlc14aDGlcOMe": ["OME", "4GA", "0GA"],
    "aDGlc13aDGalOMe": ["OME", "3LA", "0GA"],
    "aDGlc14aDGalOMe": ["OME", "4LA", "0GA"],
    "bDGlc13aDGalOMe": ["OME", "3LA", "0GB"],
    "bDGlc14aDGalOMe": ["OME", "4LA", "0GB"],
    "aDGlc16bDGalOMe": ["OME", "6LB", "0GA"],
    "bDGlc16aDGlcOMe": ["OME", "6GA", "0GB"],
    "bDGlc16bDGlcOMe": ["OME", "6GB", "0GB"],
    "aDMan12bDManOMe": ["OME", "2MB", "0MA"],
    "bDMan13bDGlcOMe": ["OME", "3GB", "0MB"],
    "aLFuc12aDManOMe": ["OME", "2MA", "0fA"],
    "aLRha12aDManOMe": ["OME", "2MA", "0hA"],
    "aLFuc(1->2)bDGal(1->3)bDGlcOMe": ["OME", "3GB", "2LB", "0fA"],
    "bLFuc(1->2)bDGal(1->3)bDGlcOMe": ["OME", "3GB", "2LB", "0fB"],
    "bDMan(1->2)aDMan(1->2)aDManOMe": ["OME", "2MA", "2MA", "0MB"],
    "bDMan(1->2)bDMan(1->2)aDManOMe": ["OME", "2MA", "2MB", "0MB"],
    "bDGlc(1->2)bDMan(1->2)aDManOMe": ["OME", "2MA", "2MB", "0GB"],
    "aDGal(1->2)aDMan(1->4)aLRhaOMe": ["OME", "4hA", "2MA", "0LA"],
    "aDMan14aLRhaOMe": ["OME", "4hA", "0MA"],
    "aDGal12aDManOMe": ["OME", "2MA", "0LA"],
    "aLRha13bDGlcOMe": ["OME", "3GB", "0hA"],
    "bLFuc12aDManOMe": ["OME", "2MA", "0fB"],
    "aDMan16aDGlcOMe": ["OME", "6GA", "0MA"],
    "bDGlc16aDGalOMe": ["OME", "6LA", "0GB"],
    "bDGal13aLRhaOMe": ["OME", "3hA", "0LB"],
    "bLFuc13aDManOMe": ["OME", "3MA", "0fB"],
    "bDMan16aDGlcOMe": ["OME", "6GA", "0MB"],
    "bDGlc16bDGalOMe": ["OME", "6LB", "0GB"],
    "aLRha13bDGalOMe": ["OME", "3LB", "0hA"],
    "bLFuc12aDGalOMe": ["OME", "2LA", "0fB"],
    "aLRha12bDGalOMe": ["OME", "2LB", "0hA"],
    "bDMan14bDGlcOMe": ["OME", "4GB", "0MB"],
    "bDGal16bDGlcOMe": ["OME", "6GB", "0LB"],
    "aLRha13aDGlcOMe": ["OME", "3GA", "0hA"],
    "aLFuc13aDGlcOMe": ["OME", "3GA", "0fA"],
    "aLFuc13bDGlcOMe": ["OME", "3GB", "0fA"],
    "aLFuc13aDGalOMe": ["OME", "3LA", "0fA"],
    "aLFuc13aDManOMe": ["OME", "3MA", "0fA"],
    "aLFuc12aDGlcOMe": ["OME", "2GA", "0fA"],
    "aLFuc14aDGlcOMe": ["OME", "4GA", "0fA"],
    "bLFuc14aDGlcOMe": ["OME", "4GA", "0fB"],
    "aDGlc14aLRhaOMe": ["OME", "4hA", "0GA"],
    "bDGlc14aLRhaOMe": ["OME", "4hA", "0GB"],
    "aLFuc14aDGalOMe": ["OME", "4LA", "0fA"],
    "bLFuc13aDGalOMe": ["OME", "3LA", "0fB"],
    "bLFuc14aDGalOMe": ["OME", "4LA", "0fB"],
    "aLFuc14aLRhaOMe": ["OME", "4hA", "0fA"],
    "bLFuc14aLRhaOMe": ["OME", "4hA", "0fB"],
    "bDGal14bDXylOMe": ["OME", "4XB", "0LB"],
    "aLFuc14aLFucOMe": ["OME", "4fA", "0fA"],
    "bLFuc14aLFucOMe": ["OME", "4fA", "0fB"],
    "aDGlc14aLFucOMe": ["OME", "4fA", "0GA"],
    "bDGlc14aLFucOMe": ["OME", "4fA", "0GB"],
    "aDGlc12bDManOMe": ["OME", "2MB", "0GA"],
    "aDGlc12aLRhaOMe": ["OME", "2hA", "0GA"],
    "aDGlc12bLRhaOMe": ["OME", "2hB", "0GA"],
    "bDGlc12aDManOMe": ["OME", "2MA", "0GB"],
    "bDGlc12bDManOMe": ["OME", "2MB", "0GB"],
    "bDGlc12aLRhaOMe": ["OME", "2hA", "0GB"],
    "bDGlc12bLRhaOMe": ["OME", "2hB", "0GB"],
    "aLFuc12bDGal": ["ROH", "2LB", "0fA"],
    "bDGal13aDGal": ["ROH", "3LA", "0LB"],
    "bDGal13bDGal": ["ROH", "3LB", "0LB"],
    "bDMan12bDGlcOMe": ["OME", "2GB", "0MB"],
    "bLFuc12bDGlcOMe": ["OME", "2GB", "0fB"],
    "bLFuc12aDGlcOMe": ["OME", "2GA", "0fB"],
    "bLFuc13aDGlcOMe": ["OME", "3GA", "0fB"],
    "aLFuc16aDGlcOMe": ["OME", "6GA", "0fA"],
    "bLFuc16aDGlcOMe": ["OME", "6GA", "0fB"],
    "aLFuc16aDGalOMe": ["OME", "6LA", "0fA"],
    "bLFuc16aDGalOMe": ["OME", "6LA", "0fB"],
    "aLFuc12aLFucOMe": ["OME", "2fA", "0fA"],
    "aLRha12aDGalOMe": ["OME", "2LA", "0hA"],
    "aLRha16aDGlc": ["ROH", "6GA", "0hA"],
    "aLRha16bDGlc": ["ROH", "6GB", "0hA"],
    "aDMan16aDMan": ["ROH", "6MA", "0MA"],
    "aDMan16bDMan": ["ROH", "6MB", "0MA"],
    "bLFuc12aLRha": ["ROH", "2hA", "0fB"],
    "bLFuc12bLRha": ["ROH", "2hB", "0fB"],
    "aLFuc12aDMan": ["ROH", "2MA", "0fA"],
    "aLFuc12bDMan": ["ROH", "2MB", "0fA"],
    "bLFuc12aDMan": ["ROH", "2MA", "0fB"],
    "bLFuc12bDMan": ["ROH", "2MB", "0fB"],
}


# GLYCAM's one-letter residue codes cover the standard pyranoses used by the
# dataset. GLYCAM uses uppercase one-letter codes for D residues and lowercase
# codes for L residues, including Fuc and Rha.
GLYCAM_BASE_CODES = {
    "DAra": "A",
    "LAra": "a",
    "DAll": "N",
    "LAll": "n",
    "DLyx": "D",
    "LLyx": "d",
    "DRibp": "R",
    "LRibp": "r",
    "DTal": "T",
    "LTal": "t",
    "DQui": "Q",
    "LQui": "q",
    "DXyl": "X",
    "LXyl": "x",
    "DGlc": "G",
    "LGlc": "g",
    "DGlcp": "G",
    "LGlcp": "g",
    "DMan": "M",
    "LMan": "m",
    "DGal": "L",
    "LGal": "l",
    "DFuc": "F",
    "LFuc": "f",
    "DRha": "H",
    "LRha": "h",
    "DGlcNAc": "Y",
    "LGlcNAc": "y",
    "DGalNAc": "V",
    "LGalNAc": "v",
    "DManNAc": "W",
    "LManNAc": "w",
    "DGlcA": "Z",
    "LGlcA": "z",
    "DGalA": "O",
    "LGalA": "o",
    "DIdoA": "U",
    "LIdoA": "u",
    "DTag": "J",
    "LTag": "j",
}
GLYCAM_BASE_NAMES = sorted(GLYCAM_BASE_CODES, key=len, reverse=True)
GLYCAM_SPECIAL_CODES = {
    # GLYCAM's Abe entries use alternate AE/AF codes for alpha/beta.
    "Abe": {"a": "AE", "b": "AF"},
    # Tyvelose is a legacy special family in GLYCAM_06j-1.prep.  Unlike the
    # regular one-letter families it uses TV/Tv for its two anomers.  This is
    # an exact Tyv mapping, not the rejected use of Tyv as a 6-deoxy-Tal
    # surrogate below.
    "Tyv": {"a": "TV", "b": "Tv"},
}


# These names are deliberately blocked rather than approximated with a
# stereochemically or chemically different GLYCAM residue.  The reasons are
# based on the residue inventory in the locally loaded GLYCAM_06j-1.prep and
# the GLYCAM residue naming table.  In particular, 0TV is tyvelose (it also
# lacks O3), 0PA/0PB are psicose, and 0BC/0bC are bacillosamine.  None is a
# safe substitute for the similarly composed rare sugars below.
GLYCAM06J_RARE_BLOCKERS: dict[str, str] = {
    "aDAllNAc": "GLYCAM06j has D-allose (N) but no N-acetyl-D-allosamine residue",
    "bDAllNAc": "GLYCAM06j has D-allose (N) but no N-acetyl-D-allosamine residue",
    "aDGlc3NAc": "GLYCAM06j Y is 2-N-acetyl-D-glucosamine; no 3-N-acetyl-D-glucose residue exists",
    "bDGlc3NAc": "GLYCAM06j Y is 2-N-acetyl-D-glucosamine; no 3-N-acetyl-D-glucose residue exists",
    "aDIdo": "GLYCAM06j has iduronic acid (U) but no neutral D-idose residue; P is psicose",
    "bDIdo": "GLYCAM06j has iduronic acid (U) but no neutral D-idose residue; P is psicose",
    "aDManA": "GLYCAM06j has GlcA (Z), GalA (O), and IdoA (U), but no D-mannuronic-acid residue",
    "aDQui4NAc": "GLYCAM06j has D-quinovose (Q) but no 4-N-acetyl-D-quinovosamine residue",
    "aDRha4NAc": "GLYCAM06j has D-rhamnose (H) but no 4-N-acetyl-D-rhamnosamine residue",
    "aDRha4NAcOMe": "GLYCAM06j has D-rhamnose (H) but no 4-N-acetyl-D-rhamnosamine residue",
    "aDRha4N": "GLYCAM06j has D-rhamnose (H) but no 4-amino-D-rhamnose residue",
    "aDRha4NOMe": "GLYCAM06j has D-rhamnose (H) but no 4-amino-D-rhamnose residue",
    "aDTal6d2NAc4NAc": (
        "GLYCAM06j has bacillosamine (0BC/0bC), not the D-talose-configured "
        "6-deoxy-2,4-di-N-acetyl target; Bac is not an allowed stereochemical surrogate"
    ),
    "aDTal6dOMe": "GLYCAM06j has D-talose (T) but no 6-deoxy-D-talose residue; 0TV is 3,6-dideoxy tyvelose",
    "aLFuc(1->2)bDGal(1->3)bD2dGlcOMe": (
        "GLYCAM06j has D-glucose (G) but no 2-deoxy-D-glucose residue for the branched acceptor"
    ),
    "aPar(1->3)[aDGal(1->2)]aDManOMe": (
        "GLYCAM06j has no paratose residue template; Abe and Tyv are distinct 3,6-dideoxy stereoisomers"
    ),
    "bDGlc2dOMe": "GLYCAM06j has D-glucose (G) but no 2-deoxy-D-glucose residue",
    "bDGlcp4dOMe": "GLYCAM06j has D-glucose (G) but no 4-deoxy-D-glucose residue",
    "bDManA": "GLYCAM06j has GlcA (Z), GalA (O), and IdoA (U), but no D-mannuronic-acid residue",
    "bDQui4NAc": "GLYCAM06j has D-quinovose (Q) but no 4-N-acetyl-D-quinovosamine residue",
    "bDTal6d2NAc4NAc": (
        "GLYCAM06j has bacillosamine (0BC/0bC), not the D-talose-configured "
        "6-deoxy-2,4-di-N-acetyl target; both Bac templates fail the project chirality gate"
    ),
    "bLFuc(1->2)bDGal(1->3)bD2dGlcOMe": (
        "GLYCAM06j has D-glucose (G) but no 2-deoxy-D-glucose residue for the branched acceptor"
    ),
}

GLYCAM06J_REJECTED_NEAR_MATCHES: dict[str, list[str]] = {
    "aDIdo": ["0PA", "0PB"],
    "bDIdo": ["0PA", "0PB"],
    "aDTal6dOMe": ["0TV", "0Tv"],
    "aDRha4N": ["0BC", "0bC"],
    "aDRha4NOMe": ["0BC", "0bC"],
    "aDTal6d2NAc4NAc": ["0BC", "0bC"],
    "aPar(1->3)[aDGal(1->2)]aDManOMe": ["0AE", "0AF", "0TV", "0Tv"],
    "bDTal6d2NAc4NAc": ["0BC", "0bC"],
}

PREP_HEADER_RE = re.compile(r"^([A-Za-z0-9]{2,4})\s+INT\s+0\s*$", re.MULTILINE)
GLYCAM_TOKEN_RE = re.compile(
    r"([ab])(" + "|".join(re.escape(name) for name in [*GLYCAM_SPECIAL_CODES, *GLYCAM_BASE_NAMES]) + r")"
)


def local_glycam_residue_names(prep_path: Path = GLYCAM_PREP) -> frozenset[str]:
    """Return the exact residue tokens provided by the loaded local prep."""
    if not prep_path.is_file():
        raise FileNotFoundError(f"GLYCAM prep not found: {prep_path}")
    names = frozenset(PREP_HEADER_RE.findall(prep_path.read_text(encoding="utf-8")))
    if not names:
        raise ValueError(f"No residue templates found in GLYCAM prep: {prep_path}")
    return names


def required_glycam_tokens(molecule: str, sequence: list[str]) -> list[str]:
    """Return all LEaP residue tokens required to build one mapped molecule."""
    required = list(sequence)
    derivative = parse_mono_derivative(molecule)
    if derivative is not None:
        derivative_code = {"Ac": "ACX", "Me": "MEX", "S": "SO3"}[derivative[1]]
        required.append(derivative_code)
    return required


def missing_local_glycam_tokens(
    molecule: str,
    sequence: list[str],
    *,
    available: frozenset[str] | set[str] | None = None,
) -> list[str]:
    """Fail-closed residue inventory check before invoking LEaP."""
    known = local_glycam_residue_names() if available is None else available
    return sorted(set(required_glycam_tokens(molecule, sequence)) - set(known))


def parse_mono_derivative(molecule: str) -> tuple[int, str, str] | None:
    """Return (position, derivative, aglycon) for supported mono derivatives."""
    tokens = list(GLYCAM_TOKEN_RE.finditer(molecule))
    if len(tokens) != 1:
        return None
    suffix = molecule[tokens[0].end() :]
    if suffix in {"p", "OMe"}:
        return None
    match = re.fullmatch(r"([2346])(Ac|S)OMe", suffix)
    if match:
        return int(match.group(1)), match.group(2), "OME"
    match = re.fullmatch(r"([2346])(Ac|S)", suffix)
    if match:
        return int(match.group(1)), match.group(2), "ROH"
    match = re.fullmatch(r"([2346])OMe", suffix)
    if match:
        return int(match.group(1)), "Me", "ROH"
    return None


def residue_code(anomer: str, base: str, *, deoxy6: bool = False) -> str | None:
    if base in GLYCAM_SPECIAL_CODES:
        return GLYCAM_SPECIAL_CODES[base][anomer]
    code = GLYCAM_BASE_CODES.get(base)
    if code is None:
        return None
    return code + ("A" if anomer == "a" else "B")


def resolve_glycam_sequence(molecule: str) -> list[str] | None:
    """Resolve simple CASPER names to a GLYCAM sequence when supported.

    Existing hand-checked sequences remain authoritative.  The generated
    path covers standard linear and two-branch glycans; nonstandard O-acetyl,
    sulfate, phosphate, unsupported N-acetyl, and unmapped residues are
    rejected before tleap so they are reported instead of silently changing
    chemistry.
    """
    if molecule in GLYCAM_SEQUENCES:
        return GLYCAM_SEQUENCES[molecule]

    tokens = [
        (match.start(), match.end(), match.group(1), match.group(2))
        for match in GLYCAM_TOKEN_RE.finditer(molecule)
    ]
    if not tokens:
        return None
    derivative = parse_mono_derivative(molecule)

    # Reject text that is not a recognized residue, linkage, branch marker,
    # or reducing-end aglycon. This catches unsupported modifications such as
    # 3NAc, phosphate, and unsupported deoxy variants.
    masked = list(molecule)
    for start, end, _, _ in tokens:
        for index in range(start, end):
            masked[index] = " "
    remainder = "".join(masked)
    remainder = re.sub(r"1(?:->)?[0-9]", "", remainder)
    remainder = remainder.replace("OMe", "").replace("[", "").replace("]", "")
    remainder = remainder.replace("(", "").replace(")", "").strip()
    if derivative is not None:
        remainder = ""
    elif len(tokens) == 1 and remainder == "p":
        remainder = ""
    if re.search(r"[A-Za-z0-9]", remainder):
        return None

    residue_codes: list[str] = []
    for _, end, anomer, base in tokens:
        suffix = molecule[end:]
        deoxy6 = bool(re.match(r"6d(?:OMe)?$", suffix))
        code = residue_code(anomer, base, deoxy6=deoxy6)
        if code is None:
            return None
        residue_codes.append(code)
    linkage_positions: list[int] = []
    for (_, end, _, _), (start, _, _, _) in zip(tokens, tokens[1:]):
        between = molecule[end:start]
        match = re.search(r"1(?:->)?([12346])", between)
        if not match:
            return None
        linkage_positions.append(int(match.group(1)))

    if derivative is not None:
        position, _, aglycon = derivative
        if len(residue_codes) != 1:
            return None
        return [aglycon, f"{position}{residue_codes[0]}"]

    aglycon = "OME" if molecule.endswith("OMe") else "ROH"
    if "[" in molecule:
        if len(tokens) != 3 or len(linkage_positions) != 2:
            return None
        linkage_set = tuple(sorted(linkage_positions))
        branch_code = {(2, 3): "Z", (3, 4): "W"}.get(linkage_set)
        if branch_code is None:
            return None
        children = [
            "0" + residue_code
            for _, residue_code in sorted(zip(linkage_positions, residue_codes[:2]))
        ]
        return [aglycon, branch_code + residue_codes[-1], *children]

    if len(linkage_positions) != len(residue_codes) - 1:
        return None
    sequence = [aglycon]
    for index in range(len(residue_codes) - 1, 0, -1):
        sequence.append(str(linkage_positions[index - 1]) + residue_codes[index])
    sequence.append("0" + residue_codes[0])
    return sequence


def resolve_glycam_sequence_with_reason(molecule: str) -> tuple[list[str] | None, str]:
    """Resolve a sequence and preserve an actionable fail-closed reason."""
    if molecule in GLYCAM06J_RARE_BLOCKERS:
        return None, GLYCAM06J_RARE_BLOCKERS[molecule]
    sequence = resolve_glycam_sequence(molecule)
    if sequence is None:
        return None, "No GLYCAM sequence mapping"
    return sequence, ""


@dataclass
class Result:
    priority: int
    molecule_name: str
    clean_name: str
    glycam_sequence: str
    run_dir: str
    tleap_status: str
    acpype_status: str
    final_status: str
    message: str


def run(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=log, stderr=subprocess.STDOUT)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return proc.returncode, text


def read_sample_rows() -> list[dict[str, str]]:
    with SAMPLE_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["priority"]))
    return rows


def branch_linkage_positions(molecule: str) -> tuple[int, int] | None:
    """Return the two acceptor positions in a CASPER two-branch name."""
    if "[" not in molecule:
        return None
    tokens = list(GLYCAM_TOKEN_RE.finditer(molecule))
    positions: list[int] = []
    for left, right in zip(tokens, tokens[1:]):
        between = molecule[left.end() : right.start()]
        match = re.search(r"1(?:->)?([2346])", between)
        if not match:
            return None
        positions.append(int(match.group(1)))
    if len(tokens) != 3 or len(positions) != 2:
        return None
    return tuple(sorted(positions))


def write_leap_input(path: Path, molecule: str, sequence: list[str], stem: str) -> None:
    branch_positions = branch_linkage_positions(molecule)
    if len(sequence) == 3 and sequence[1].startswith("1") and sequence[2].startswith("0"):
        # Anomeric-anomeric (1->1) linkages need one residue with an open O1
        # and one with an open anomeric carbon. A linear sequence would leave
        # an extra reducing-end hydroxyl and generate invalid torsions.
        first_residue = "1" + sequence[2][1:]
        second_residue = "0" + sequence[1][1:]
        text = "\n".join(
            [
                "source leaprc.GLYCAM_06j-1",
                "",
                f"g1 = sequence {{ {first_residue} }}",
                f"g2 = sequence {{ {second_residue} }}",
                "mol = combine { g1 g2 }",
                "bond mol.1.O1 mol.2.C1",
                "check mol",
                "charge mol",
                f"saveamberparm mol {stem}.prmtop {stem}.inpcrd",
                f"savepdb mol {stem}_leap.pdb",
                "quit",
                "",
            ]
        )
    elif branch_positions is not None:
        # A Z/W residue encodes the branching central residue, but LEaP's
        # sequence parser cannot reliably infer both branch bonds from three
        # linear residue tokens. Build each branch as a separate unit and add
        # the two glycosidic bonds explicitly.
        aglycon, branch_residue, *children = sequence
        lines = [
            "source leaprc.GLYCAM_06j-1",
            "",
            f"base = sequence {{ {aglycon} {branch_residue} }}",
        ]
        for index, child in enumerate(children, start=1):
            lines.append(f"branch{index} = sequence {{ {child} }}")
        units = " ".join(["base"] + [f"branch{index}" for index in range(1, len(children) + 1)])
        lines.append(f"mol = combine {{ {units} }}")
        central_residue = 2
        for index, position in enumerate(branch_positions, start=3):
            lines.append(f"bond mol.{central_residue}.O{position} mol.{index}.C1")
        lines.extend(
            [
                "check mol",
                "charge mol",
                f"saveamberparm mol {stem}.prmtop {stem}.inpcrd",
                f"savepdb mol {stem}_leap.pdb",
                "quit",
                "",
            ]
        )
        text = "\n".join(lines)
    elif (derivative := parse_mono_derivative(molecule)) is not None:
        position, derivative_name, _ = derivative
        derivative_code = {"Ac": "ACX", "Me": "MEX", "S": "SO3"}[derivative_name]
        text = "\n".join(
            [
                "source leaprc.GLYCAM_06j-1",
                "",
                f"mol = sequence {{ {' '.join(sequence)} }}",
                f"set mol tail mol.2.O{position}",
                f"mol = sequence {{ mol {derivative_code} }}",
                "check mol",
                "charge mol",
                f"saveamberparm mol {stem}.prmtop {stem}.inpcrd",
                f"savepdb mol {stem}_leap.pdb",
                "quit",
                "",
            ]
        )
    else:
        text = "\n".join(
            [
                "source leaprc.GLYCAM_06j-1",
                "",
                f"mol = sequence {{ {' '.join(sequence)} }}",
                "check mol",
                "charge mol",
                f"saveamberparm mol {stem}.prmtop {stem}.inpcrd",
                f"savepdb mol {stem}_leap.pdb",
                "quit",
                "",
            ]
        )
    path.write_text(text, encoding="utf-8")


def run_one(row: dict[str, str], *, run_acpype: bool) -> Result:
    molecule = row["molecule_name"]
    clean = row["clean_name"]
    priority = int(row["priority"])
    sequence, mapping_reason = resolve_glycam_sequence_with_reason(molecule)
    run_dir = RUN_ROOT / clean / "glycam06j_topology_dryrun"
    run_dir.mkdir(parents=True, exist_ok=True)

    if sequence is None:
        available_templates: frozenset[str] = frozenset()
        if GLYCAM_PREP.is_file():
            available_templates = local_glycam_residue_names()
        evidence = {
            "schema_version": 2,
            "molecule_name": molecule,
            "clean_name": clean,
            "mapping_status": "unsupported",
            "reason": mapping_reason,
            "force_field": "GLYCAM06j",
            "leaprc": str(GLYCAM_LEAPRC),
            "prep": str(GLYCAM_PREP),
            "prep_template_count": len(available_templates),
            "rejected_near_match_tokens": GLYCAM06J_REJECTED_NEAR_MATCHES.get(molecule, []),
            "policy": {
                "gaff_fallback": "forbidden",
                "stereoisomer_substitution": "forbidden",
                "unvalidated_custom_charges": "forbidden",
                "derivative_rule": (
                    "ACX is an O-acetyl derivative and cannot replace an OH with an N-acetyl amino group"
                ),
                "production_eligibility": "blocked until an exact official GLYCAM template is available",
            },
        }
        if GLYCAM_LEAPRC.is_file():
            evidence["leaprc_sha256"] = sha256(GLYCAM_LEAPRC)
        if GLYCAM_PREP.is_file():
            evidence["prep_sha256"] = sha256(GLYCAM_PREP)
        (run_dir / "glycam06j_mapping_blocker.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return Result(
            priority,
            molecule,
            clean,
            "",
            str(run_dir),
            "skipped",
            "skipped",
            "failed",
            mapping_reason,
        )

    if not GLYCAM_LEAPRC.is_file():
        return Result(
            priority,
            molecule,
            clean,
            " ".join(sequence),
            str(run_dir),
            "skipped",
            "skipped",
            "failed",
            f"Local GLYCAM leaprc unavailable: {GLYCAM_LEAPRC}",
        )
    try:
        missing_tokens = missing_local_glycam_tokens(molecule, sequence)
    except (FileNotFoundError, ValueError) as exc:
        return Result(
            priority,
            molecule,
            clean,
            " ".join(sequence),
            str(run_dir),
            "skipped",
            "skipped",
            "failed",
            f"Local GLYCAM residue inventory unavailable: {exc}",
        )
    if missing_tokens:
        return Result(
            priority,
            molecule,
            clean,
            " ".join(sequence),
            str(run_dir),
            "skipped",
            "skipped",
            "failed",
            "Local GLYCAM06j residue template missing: " + ", ".join(missing_tokens),
        )

    stem = f"{clean}_GLYCAM06j"
    leap_input = run_dir / "build_glycam06j.leap"
    write_leap_input(leap_input, molecule, sequence, stem)
    metadata = {
        "molecule_name": molecule,
        "clean_name": clean,
        "priority": priority,
        "glycam_sequence": sequence,
        "leap_input": str(leap_input),
        "source_sample_csv": str(SAMPLE_CSV),
        "glycam_library_evidence": {
            "leaprc": str(GLYCAM_LEAPRC),
            "leaprc_sha256": sha256(GLYCAM_LEAPRC),
            "prep": str(GLYCAM_PREP),
            "prep_sha256": sha256(GLYCAM_PREP),
            "validated_residue_tokens": required_glycam_tokens(molecule, sequence),
        },
    }
    (run_dir / "glycam06j_topology_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    prmtop = run_dir / f"{stem}.prmtop"
    inpcrd = run_dir / f"{stem}.inpcrd"
    if prmtop.exists() and inpcrd.exists():
        tleap_status = "done"
        tleap_msg = "existing prmtop/inpcrd"
    else:
        rc, log = run([str(TLEAP), "-f", leap_input.name], run_dir, run_dir / "tleap.log")
        tleap_status = "done" if rc == 0 and prmtop.exists() and inpcrd.exists() else "failed"
        tleap_msg = "tleap ok" if tleap_status == "done" else tail(log)
    if tleap_status != "done":
        return Result(priority, molecule, clean, " ".join(sequence), str(run_dir), tleap_status, "skipped", "failed", tleap_msg)

    amb2gmx_dir = run_dir / f"{stem}.amb2gmx"
    if not run_acpype:
        return Result(priority, molecule, clean, " ".join(sequence), str(run_dir), tleap_status, "not_run", "prepared", "tleap topology generated")
    gromacs_top = amb2gmx_dir / f"{stem}_GMX.top"
    gromacs_gro = amb2gmx_dir / f"{stem}_GMX.gro"
    if gromacs_top.exists() and gromacs_gro.exists():
        acpype_status = "done"
        acpype_message = "existing amb2gmx topology"
    else:
        rc, log = run([str(ACPYPE), "-p", prmtop.name, "-x", inpcrd.name, "-b", stem], run_dir, run_dir / "acpype.log")
        acpype_status = "done" if rc == 0 and gromacs_top.exists() and gromacs_gro.exists() else "failed"
        acpype_message = "acpype ok" if acpype_status == "done" else tail(log)
    if acpype_status != "done":
        return Result(priority, molecule, clean, " ".join(sequence), str(run_dir), tleap_status, acpype_status, "failed", acpype_message)

    coordinate_metadata_path = run_dir / "glycam_coordinate_repair_metadata.json"
    chirality_audit_path = run_dir / "glycam_chirality_audit.json"
    try:
        repair_metadata = repair_coordinates(
            molecule_name=molecule,
            source_sdf=Path(row["source_sdf"]),
            mmff_pickle=Path(row["mmff100_pickle"]),
            gromacs_top=gromacs_top,
            gromacs_gro=gromacs_gro,
            atom_map_output=run_dir / "sdf_to_glycam_frame_heavy_index_map.json",
            metadata_output=coordinate_metadata_path,
        )
        raw_gro = gromacs_gro.with_name(gromacs_gro.stem + "_tleap_raw.gro")
        chirality_audit = audit_chirality(
            topology=gromacs_top,
            reference_gro=raw_gro,
            candidate_gro=gromacs_gro,
        )
        chirality_audit_path.write_text(
            json.dumps(chirality_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not chirality_audit["strict_passed"]:
            raise ValueError(
                "raw-to-repaired chirality gate failed: "
                f"mismatch={chirality_audit['counts']['mismatched']} "
                f"unevaluable={chirality_audit['counts']['unevaluable']}"
            )
        repair_metadata["chirality_audit"] = {
            "strict_passed": True,
            "counts": chirality_audit["counts"],
            "audit_file": str(chirality_audit_path),
        }
        coordinate_metadata_path.write_text(
            json.dumps(repair_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        return Result(
            priority,
            molecule,
            clean,
            " ".join(sequence),
            str(run_dir),
            tleap_status,
            acpype_status,
            "failed",
            f"coordinate repair failed: {exc}",
        )
    return Result(
        priority,
        molecule,
        clean,
        " ".join(sequence),
        str(run_dir),
        tleap_status,
        acpype_status,
        "prepared",
        acpype_message + "; MMFF start geometry mapped",
    )


def tail(text: str, n: int = 12) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines[-n:])


def write_results(results: list[Result]) -> None:
    status_csv = RUN_ROOT / "extended24_glycam06j_topology_status.csv"
    status_json = RUN_ROOT / "extended24_glycam06j_topology_status.json"
    report = RUN_ROOT / "extended24_glycam06j_topology_status.md"
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [result.__dict__ for result in results]
    with status_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    status_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prepared = sum(1 for result in results if result.final_status == "prepared")
    failed = sum(1 for result in results if result.final_status == "failed")
    lines = [
        "# Extended-24 GLYCAM06j topology dry-run status",
        "",
        f"- prepared: {prepared}",
        f"- failed: {failed}",
        "",
        "| priority | molecule | GLYCAM sequence | tleap | acpype | final |",
        "|---:|---|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.priority} | `{result.molecule_name}` | `{result.glycam_sequence}` | {result.tleap_status} | {result.acpype_status} | {result.final_status} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GLYCAM06j topologies for selected extended glycans.")
    parser.add_argument("--no-acpype", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Restrict to a clean_name or molecule_name; repeat for multiple glycans.",
    )
    parser.add_argument(
        "--remaining-only",
        action="store_true",
        help="Prepare only glycans missing either the all-200 or low-100 GLYCAM output pickle.",
    )
    parser.add_argument(
        "--skip-existing-topology",
        action="store_true",
        help="Skip only complete topology directories; reject partial existing directories.",
    )
    args = parser.parse_args()
    if not TLEAP.exists():
        raise SystemExit(f"tleap not found: {TLEAP}")
    if not ACPYPE.exists():
        raise SystemExit(f"acpype not found: {ACPYPE}")
    if not GLYCAM_LEAPRC.is_file():
        raise SystemExit(f"GLYCAM leaprc not found: {GLYCAM_LEAPRC}")
    if not GLYCAM_PREP.is_file():
        raise SystemExit(f"GLYCAM prep not found: {GLYCAM_PREP}")
    shutil.which("python")
    rows = read_sample_rows()
    if args.remaining_only:
        all_dir = MD_ROOT / "conformations_md_glycam06j_water_5ns_200cand_all_seed1"
        low_dir = MD_ROOT / "conformations_md_glycam06j_water_5ns_200cand_lowest100_sugar_energy_seed1"
        rows = [
            row
            for row in rows
            if not (all_dir / f"{row['clean_name']}.pickle").exists()
            or not (low_dir / f"{row['clean_name']}.pickle").exists()
        ]
    if args.only:
        selected = set(args.only)
        rows = [row for row in rows if row["clean_name"] in selected or row["molecule_name"] in selected]
        if not rows:
            raise SystemExit("No sample matched --only")
    if args.skip_existing_topology:
        pending_rows: list[dict[str, str]] = []
        for row in rows:
            clean = row["clean_name"]
            stem = f"{clean}_GLYCAM06j"
            run_dir = RUN_ROOT / clean / "glycam06j_topology_dryrun"
            acpype_dir = run_dir / f"{stem}.amb2gmx"
            required = (
                acpype_dir / f"{stem}_GMX.gro",
                acpype_dir / f"{stem}_GMX.top",
                acpype_dir / f"{stem}_GMX_tleap_raw.gro",
                run_dir / "sdf_to_glycam_frame_heavy_index_map.json",
                run_dir / "glycam_coordinate_repair_metadata.json",
            )
            if not run_dir.exists():
                pending_rows.append(row)
            elif all(path.exists() for path in required):
                continue
            else:
                raise SystemExit(
                    f"refusing partial existing topology directory with --skip-existing-topology: {run_dir}"
                )
        rows = pending_rows
        if not rows:
            print(json.dumps({"prepared": 0, "failed": 0, "run_root": str(RUN_ROOT)}, ensure_ascii=False, indent=2))
            return
    results = [run_one(row, run_acpype=not args.no_acpype) for row in rows]
    write_results(results)
    prepared = sum(1 for result in results if result.final_status == "prepared")
    failed = sum(1 for result in results if result.final_status == "failed")
    print(json.dumps({"prepared": prepared, "failed": failed, "run_root": str(RUN_ROOT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
