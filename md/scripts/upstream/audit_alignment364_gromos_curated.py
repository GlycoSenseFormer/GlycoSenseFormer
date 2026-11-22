#!/usr/bin/env python3
"""Audit the 19 alignment-364 GROMOS gaps against curated local assets.

This script is deliberately fail-closed.  It does not fall back to the
graph-derived XGO builder and it does not schedule MD.  A future curated
entry must pass :mod:`gromos_curated_gate` before it can be reported as
production ready.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from gromos_curated_gate import REQUIRED_STRICT_CHECKS, metadata_is_production_ready


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
DEFAULT_MANIFEST = MD_ROOT / "md_alignment364_v1" / "alignment364_targets.csv"
DEFAULT_OUTPUT_ROOT = MD_ROOT / "md_alignment364_v1" / "gromos"
SOURCE_FF = (
    MD_ROOT
    / "forcefield_assets"
    / "gromos56a6carbo"
    / "gromos53a6carbo.ff"
)
RTP_PATH = SOURCE_FF / "aminoacids.rtp"
README_PATH = SOURCE_FF.parent / "README.txt"

EXPECTED_GROMOS_TARGETS = frozenset(
    {
        "aDGal2SOMe",
        "aDGal3SOMe",
        "aDGal4SOMe",
        "aDGal6SOMe",
        "aDGlc6S",
        "aDMan",
        "aDRha4N",
        "aDRha4NOMe",
        "aDTal6d2NAc4NAc",
        "aLFuc12[aDGal13]bDGal",
        "bDGal14aDXyl",
        "bDGlc6S",
        "bDGlcNAc",
        "bDGlcNAc13bDGal",
        "bDManA",
        "bDQui",
        "bDQui4NAc",
        "bLFuc",
        "bLRha",
    }
)

# No target is currently approved.  New entries must include a builder and a
# complete strict-validation metadata record; adding only a name is not enough.
CURATED_TARGET_METADATA: dict[str, dict[str, object]] = {}


def _blocker(
    code: str,
    detail: str,
    *,
    candidates: tuple[str, ...] = (),
    missing: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "blocker_code": code,
        "blocker_detail": detail,
        "candidate_templates": list(candidates),
        "missing_curated_components": list(missing),
    }


BLOCKERS: dict[str, dict[str, object]] = {}
for _name in ("aDGal2SOMe", "aDGal3SOMe", "aDGal4SOMe", "aDGal6SOMe"):
    BLOCKERS[_name] = _blocker(
        "missing_sulfated_methyl_galactose_template",
        "GALA/GALB are 1->4 polymer residues and the local force field has no positional sulfate or O-methyl terminal patch.",
        candidates=("GALA", "GALB"),
        missing=("position-specific sulfate patch", "methyl-glycoside terminal template"),
    )
for _name in ("aDGlc6S", "bDGlc6S"):
    BLOCKERS[_name] = _blocker(
        "missing_glucose_6_sulfate_patch",
        "The curated glucose entries contain no sulfur and no O6-sulfation patch is present.",
        candidates=("GLCA", "GLCB"),
        missing=("O6-sulfate residue/patch",),
    )
BLOCKERS.update(
    {
        "aDMan": _blocker(
            "mana_is_not_a_free_monosaccharide_template",
            "MANA/MANB contain cross-residue +/- references for a 1->4 chain and omit the free O4/HO4 and HO1 termini; GLCB is free beta-D-glucose and has the wrong C2/anomer stereochemistry.",
            candidates=("MANA", "MANB", "GLCB"),
            missing=("curated free alpha-D-mannose template",),
        ),
        "aDRha4N": _blocker(
            "missing_4_amino_rhamnose_template",
            "No rhamnose or 4-amino-rhamnose residue exists in the local curated RTP.",
            missing=("alpha-D-rhamnose-4-amine residue",),
        ),
        "aDRha4NOMe": _blocker(
            "missing_4_amino_rhamnose_methyl_template",
            "No rhamnose/4-amino-rhamnose residue or methyl-glycoside terminal patch exists.",
            missing=("alpha-D-rhamnose-4-amine residue", "methyl-glycoside terminal patch"),
        ),
        "aDTal6d2NAc4NAc": _blocker(
            "missing_dideoxy_dinacetyl_talose_template",
            "The local curated RTP has no 6-deoxy-talose 2,4-di-N-acetyl residue.",
            missing=("6-deoxy-D-talose-2,4-di-N-acetyl residue",),
        ),
        "aLFuc12[aDGal13]bDGal": _blocker(
            "unsupported_residue_and_linkages",
            "LFuc is absent and the distributed 56A_CARBO workflow explicitly handles only 1->4, not the target 1->2/1->3 branch.",
            candidates=("GALA", "GALB"),
            missing=("alpha-L-fucose residue", "1->2 linkage patch", "1->3 linkage patch", "branch patch"),
        ),
        "bDGal14aDXyl": _blocker(
            "missing_xylose_residue",
            "The 1->4 linkage class is supported, but no alpha-D-xylose curated residue is present.",
            candidates=("GALA", "GALB"),
            missing=("alpha-D-xylose reducing-end residue",),
        ),
        "bDGlcNAc": _blocker(
            "missing_glcnac_template",
            "GLCN denotes a glucose reducing-end building block in this distribution, not glucosamine or GlcNAc; no nitrogen-containing carbohydrate template is present.",
            candidates=("GLCN", "GLCB"),
            missing=("free beta-D-GlcNAc residue",),
        ),
        "bDGlcNAc13bDGal": _blocker(
            "missing_glcnac_and_1to3_patch",
            "No GlcNAc residue is present and the distributed workflow does not curate a 1->3 linkage.",
            candidates=("GLCN", "GALA", "GALB"),
            missing=("beta-D-GlcNAc residue", "1->3 linkage patch"),
        ),
        "bDManA": _blocker(
            "missing_mannuronic_acid_template",
            "MANA/MANB are neutral 1->4 mannose chain residues, not beta-D-mannuronic acid.",
            candidates=("MANA", "MANB"),
            missing=("beta-D-mannuronic-acid residue",),
        ),
        "bDQui": _blocker(
            "missing_quinovose_template",
            "No 6-deoxy-D-glucose/quinovose curated residue exists in the local RTP.",
            missing=("free beta-D-quinovose residue",),
        ),
        "bDQui4NAc": _blocker(
            "missing_quinovose_4_nacetyl_template",
            "No quinovose or 4-N-acetyl-quinovose curated residue exists.",
            missing=("beta-D-quinovose-4-N-acetyl residue",),
        ),
        "bLFuc": _blocker(
            "missing_l_fucose_template",
            "No free beta-L-fucose curated residue exists in the local RTP.",
            missing=("free beta-L-fucose residue",),
        ),
        "bLRha": _blocker(
            "missing_l_rhamnose_template",
            "No free beta-L-rhamnose curated residue exists in the local RTP.",
            missing=("free beta-L-rhamnose residue",),
        ),
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_gromos_rows(manifest: Path = DEFAULT_MANIFEST) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if "gromos" in row["missing_forcefields"].split(";")
        ]
    observed = {row["clean_name"] for row in rows}
    if len(rows) != 19 or observed != EXPECTED_GROMOS_TARGETS:
        raise ValueError(
            "frozen alignment manifest GROMOS set changed: "
            f"count={len(rows)} missing={sorted(EXPECTED_GROMOS_TARGETS - observed)} "
            f"extra={sorted(observed - EXPECTED_GROMOS_TARGETS)}"
        )
    rows.sort(key=lambda row: int(row["alignment_priority"]))
    return rows


def parse_rtp_templates(path: Path = RTP_PATH) -> dict[str, dict[str, object]]:
    """Parse enough RTP structure to distinguish free and chain residues."""

    templates: dict[str, dict[str, object]] = {}
    current_name: str | None = None
    current_section: str | None = None
    top_level = re.compile(r"^\[\s*([^\]]+?)\s*\]\s*$")
    subsection = re.compile(r"^\s+\[\s*([^\]]+?)\s*\]\s*$")
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        top_match = top_level.match(raw_line)
        if top_match:
            current_name = top_match.group(1)
            templates[current_name] = {
                "atom_names": [],
                "atom_types": [],
                "external_references": set(),
            }
            current_section = None
            continue
        sub_match = subsection.match(raw_line)
        if sub_match and current_name is not None:
            current_section = sub_match.group(1).lower()
            continue
        if current_name is None:
            continue
        stripped = raw_line.split(";", 1)[0].strip()
        if not stripped:
            continue
        tokens = stripped.split()
        template = templates[current_name]
        if current_section == "atoms" and len(tokens) >= 2:
            template["atom_names"].append(tokens[0])
            template["atom_types"].append(tokens[1])
        for token in tokens:
            if re.fullmatch(r"[+-][A-Za-z][A-Za-z0-9]*", token):
                template["external_references"].add(token)
    for template in templates.values():
        template["external_references"] = sorted(template["external_references"])
        template["atom_count"] = len(template["atom_names"])
    return templates


def installed_gromos_asset_scan() -> list[dict[str, object]]:
    """Inventory carbohydrate-like entries in the installed GROMACS GROMOS sets."""

    executable = shutil.which("gmx")
    if executable is None:
        return []
    top_root = Path(executable).resolve().parents[1] / "share" / "gromacs" / "top"
    if not top_root.is_dir():
        return []
    relevant = re.compile(r"^(?:GLC|GAL|MAN|FUC|RHA|QUI|XYL)")
    results: list[dict[str, object]] = []
    for rtp in sorted(top_root.glob("gromos*.ff/aminoacids.rtp")):
        templates = parse_rtp_templates(rtp)
        names = sorted(name for name in templates if relevant.match(name))
        patch_hits: list[dict[str, object]] = []
        for patch_path in sorted(rtp.parent.glob("*.tdb")):
            hits = sorted(
                {
                    match.group(0)
                    for match in re.finditer(
                        r"\b(?:GLC[A-Z0-9]*|GAL[A-Z0-9]*|MAN[A-Z0-9]*|FUC[A-Z0-9]*|RHA[A-Z0-9]*|QUI[A-Z0-9]*|XYL[A-Z0-9]*)\b",
                        patch_path.read_text(encoding="utf-8", errors="replace"),
                    )
                }
            )
            if hits:
                patch_hits.append({"path": str(patch_path.resolve()), "hits": hits})
        results.append(
            {
                "forcefield": rtp.parent.name,
                "rtp": str(rtp.resolve()),
                "rtp_sha256": sha256_file(rtp),
                "carbohydrate_like_templates": names,
                "carbohydrate_patch_hits": patch_hits,
                "mana": templates.get("MANA"),
                "manb": templates.get("MANB"),
            }
        )
    return results


def _source_molecule(row: dict[str, str]) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(row["source_sdf"], removeHs=False)
    molecule = supplier[int(row["source_index"])]
    if molecule is None or not molecule.HasProp("_Name"):
        raise ValueError(f"cannot load source identity for {row['clean_name']}")
    if molecule.GetProp("_Name") != row["molecule_name"]:
        raise ValueError(
            f"source identity mismatch for {row['clean_name']}: "
            f"{molecule.GetProp('_Name')!r} != {row['molecule_name']!r}"
        )
    if molecule.GetNumHeavyAtoms() != int(row["n_heavy_atoms"]):
        raise ValueError(f"heavy-atom count mismatch for {row['clean_name']}")
    return molecule


def build_audit(
    manifest: Path = DEFAULT_MANIFEST,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, dict[str, object]]]:
    rows = read_gromos_rows(manifest)
    if set(BLOCKERS) != EXPECTED_GROMOS_TARGETS:
        raise ValueError("blocker catalog must cover the frozen 19-target set exactly")
    templates = parse_rtp_templates()
    required_templates = {"GLCA", "GLCB", "GLCN", "GALA", "GALB", "MANA", "MANB"}
    if not required_templates.issubset(templates):
        raise ValueError(
            f"curated RTP candidates missing: {sorted(required_templates - set(templates))}"
        )

    records: list[dict[str, object]] = []
    for row in rows:
        clean = row["clean_name"]
        molecule = _source_molecule(row)
        metadata = CURATED_TARGET_METADATA.get(clean)
        ready = metadata_is_production_ready(metadata) if metadata is not None else False
        if ready:
            status = "production_ready"
            blocker = {
                "blocker_code": "",
                "blocker_detail": "",
                "candidate_templates": metadata.get("curated_template_ids", []),
                "missing_curated_components": [],
            }
        else:
            status = "blocked_fail_closed"
            blocker = BLOCKERS[clean]
        element_counts = Counter(atom.GetSymbol() for atom in molecule.GetAtoms())
        record: dict[str, object] = {
            "alignment_priority": int(row["alignment_priority"]),
            "source_index": int(row["source_index"]),
            "molecule_name": row["molecule_name"],
            "clean_name": clean,
            "source_class": row["source_class"],
            "source_sdf": row["source_sdf"],
            "source_sdf_sha256": row["source_sdf_sha256"],
            "baseline_region": row["baseline_region"],
            "n_heavy_atoms": molecule.GetNumHeavyAtoms(),
            "formula": rdMolDescriptors.CalcMolFormula(molecule),
            "element_counts": dict(sorted(element_counts.items())),
            "formal_charge": sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
            "parameterization_tier": (
                metadata.get("parameterization_tier") if ready and metadata else "unavailable"
            ),
            "status": status,
            "production_ready": ready,
            "topology_dryrun_allowed": ready,
            "short_em_nvt_allowed": ready,
            **blocker,
        }
        records.append(record)

    ready_count = sum(record["production_ready"] is True for record in records)
    candidate_inventory = {
        name: templates[name]
        for name in sorted(required_templates)
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "scope": "alignment364_gromos_missing19_curated_audit",
        "frozen_manifest": str(manifest.resolve()),
        "frozen_manifest_sha256": sha256_file(manifest),
        "source_forcefield": str(SOURCE_FF.resolve()),
        "source_rtp_sha256": sha256_file(RTP_PATH),
        "source_readme_sha256": sha256_file(README_PATH),
        "target_count": len(records),
        "production_ready_count": ready_count,
        "blocked_count": len(records) - ready_count,
        "topology_dryruns_started": 0,
        "short_em_nvt_started": 0,
        "five_ns_md_started": 0,
        "formal_alignment_increment": ready_count,
        "required_strict_checks": list(REQUIRED_STRICT_CHECKS),
        "scientific_boundary": (
            "RTP name similarity, grompp success, EM, or a short NVT alone cannot establish parameter validity."
        ),
        "asset_findings": {
            "distributed_workflow_linkage_scope": "1->4 only",
            "readme_new_carbohydrate_entries": ["GLCB", "GLC0", "GLC", "GLCN"],
            "glcn_meaning": "glucose reducing end; not GlcN or GlcNAc",
            "mana_manb_scope": "1->4 chain residues with external +/- atom references; not free monosaccharides",
        },
        "installed_gromos_asset_scan": installed_gromos_asset_scan(),
    }
    return records, summary, candidate_inventory


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
    finally:
        temporary.unlink(missing_ok=True)


def _csv_payload(rows: list[dict[str, object]]) -> bytes:
    with tempfile.TemporaryDirectory() as temporary_name:
        staged = Path(temporary_name) / "audit.csv"
        with staged.open("w", encoding="utf-8", newline="") as handle:
            fields = list(rows[0])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                flat = dict(row)
                for key in ("element_counts", "candidate_templates", "missing_curated_components"):
                    if key in flat:
                        flat[key] = json.dumps(flat[key], ensure_ascii=False, sort_keys=True)
                writer.writerow(flat)
        return staged.read_bytes()


def write_audit(
    manifest: Path = DEFAULT_MANIFEST,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> list[Path]:
    records, summary, inventory = build_audit(manifest)
    targets = read_gromos_rows(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    targets_path = output_root / "alignment364_gromos_missing19.csv"
    audit_csv = output_root / "curated_audit.csv"
    audit_json = output_root / "curated_audit.json"
    summary_json = output_root / "curated_audit_summary.json"
    inventory_json = output_root / "curated_template_inventory.json"
    report_md = output_root / "CURATED_AUDIT.md"

    _atomic_write(targets_path, _csv_payload([dict(row) for row in targets]))
    _atomic_write(audit_csv, _csv_payload(records))
    _atomic_write(
        audit_json,
        (json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    _atomic_write(
        summary_json,
        (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    _atomic_write(
        inventory_json,
        (json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )

    lines = [
        "# Alignment-364 GROMOS curated audit",
        "",
        f"- Frozen missing-GROMOS targets: **{summary['target_count']}**",
        f"- Production-ready from official/local curated assets: **{summary['production_ready_count']}**",
        f"- Fail-closed blockers: **{summary['blocked_count']}**",
        "- Topology dry-runs / short EM-NVT / 5 ns started: **0 / 0 / 0**",
        "",
        "`MANA`/`MANB` are not free mannose templates: both contain cross-residue references and omit free-terminal atoms. `GLCN` is the glucose reducing-end residue named by the distribution README, not GlcN/GlcNAc.",
        "",
        "| priority | target | status | candidate templates | blocker |",
        "|---:|---|---|---|---|",
    ]
    for record in records:
        candidates = ", ".join(record["candidate_templates"]) or "-"
        lines.append(
            f"| {record['alignment_priority']} | `{record['clean_name']}` | "
            f"{record['status']} | `{candidates}` | {record['blocker_code']} |"
        )
    lines.extend(
        [
            "",
            "## Release rule",
            "",
            "A future topology may set `production_ready=true` only after official/local curated provenance, a source SHA256, explicit template IDs, and all seven strict checks are present. Graph-derived `XGO` topologies remain diagnostic even if grompp, EM, or NVT succeeds.",
        ]
    )
    _atomic_write(report_md, ("\n".join(lines) + "\n").encode("utf-8"))
    return [targets_path, audit_csv, audit_json, summary_json, inventory_json, report_md]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    for path in write_audit(args.manifest, args.output_root):
        print(path)


if __name__ == "__main__":
    main()
