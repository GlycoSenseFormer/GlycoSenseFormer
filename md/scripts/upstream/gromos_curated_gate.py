#!/usr/bin/env python3
"""Fail-closed production-readiness gate for GROMOS carbohydrate topologies.

Numerical preprocessing or a short smoke trajectory is not parameter
provenance.  A topology may be marked production ready only when it was
built from an official or locally curated source and all identity and short
simulation checks were recorded explicitly.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path


CURATED_PARAMETER_TIERS = frozenset({"official_standard", "locally_curated"})
CURATED_PROVENANCE = frozenset(
    {
        "official_forcefield_template",
        "local_curated_forcefield_template",
    }
)
REQUIRED_STRICT_CHECKS = (
    "source_identity",
    "atom_identity",
    "stereochemistry",
    "formal_charge",
    "topology_preprocess",
    "energy_minimization",
    "short_nvt",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
DEFAULT_APPROVED_MANIFEST = (
    MD_ROOT
    / "md_alignment364_v1"
    / "gromos"
    / "approved_p3_production_manifest_v1.json"
)
P3_REQUIRED_GATES = {
    "official_source_licence": "PASS",
    "isolated_build": "PASS",
    "five_conformer_energy_equivalence": "PASS",
    "five_conformer_total_force_equivalence": "PASS",
    "identity_atom_mapping_17_of_17": "PASS",
    "strict_grompp_em": "PASS_MAXWARN_0",
    "strict_grompp_nvt": "PASS_MAXWARN_0",
    "diagnostic_em": "PASS",
    "diagnostic_nvt_0_5_ps": "PASS",
    "five_ns_production": "NOT_RUN_PROHIBITED",
}
APPROVED_MANIFEST_SCHEMA = "gromos_p3_approved_production_manifest/v1"


def production_readiness_errors(metadata: Mapping[str, object]) -> list[str]:
    """Return every reason that a metadata object is not production ready."""

    errors: list[str] = []
    tier = metadata.get("parameterization_tier")
    if tier not in CURATED_PARAMETER_TIERS:
        errors.append(
            "parameterization_tier must be official_standard or locally_curated"
        )

    provenance = metadata.get("topology_provenance")
    if provenance not in CURATED_PROVENANCE:
        errors.append("topology_provenance is not an approved curated source")

    template_ids = metadata.get("curated_template_ids")
    if not isinstance(template_ids, list) or not template_ids or not all(
        isinstance(item, str) and item.strip() for item in template_ids
    ):
        errors.append("curated_template_ids must contain at least one template")

    source_sha256 = metadata.get("curated_source_sha256")
    if not isinstance(source_sha256, str) or SHA256_RE.fullmatch(source_sha256) is None:
        errors.append("curated_source_sha256 must be a lowercase SHA256 digest")

    if metadata.get("experimental_gromos") is True:
        errors.append("experimental_gromos cannot be production ready")
    if metadata.get("graph_derived_parameters") is True:
        errors.append("graph-derived parameters cannot be production ready")

    strict = metadata.get("strict_validation")
    if not isinstance(strict, Mapping):
        errors.append("strict_validation object is missing")
        return errors
    if strict.get("passed") is not True:
        errors.append("strict_validation.passed is not true")
    checks = strict.get("checks")
    if not isinstance(checks, Mapping):
        errors.append("strict_validation.checks object is missing")
        return errors
    for check in REQUIRED_STRICT_CHECKS:
        if checks.get(check) is not True:
            errors.append(f"strict validation check not passed: {check}")
    return errors


def metadata_is_production_ready(metadata: Mapping[str, object]) -> bool:
    return metadata.get("production_ready") is True and not production_readiness_errors(
        metadata
    )


def validate_production_ready_claim(metadata: Mapping[str, object]) -> None:
    """Reject a forged or incomplete ``production_ready=true`` claim."""

    if metadata.get("production_ready") is not True:
        return
    errors = production_readiness_errors(metadata)
    if errors:
        raise RuntimeError(
            "invalid GROMOS production_ready claim: " + " | ".join(errors)
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory_tree(path: Path) -> dict[str, object]:
    """Hash a frozen input tree including relative names and file sizes."""

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if child.is_symlink():
            raise RuntimeError(f"approved GROMOS input tree contains a symlink: {child}")
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix()
        size = child.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
        file_count += 1
        total_bytes += size
    if file_count == 0:
        raise RuntimeError(f"approved GROMOS input tree is empty: {path}")
    return {
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def _verify_bound_file(record: object, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{label} provenance record is missing")
    raw_path = record.get("path")
    expected = record.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"{label} path is missing")
    if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
        raise RuntimeError(f"{label} SHA256 is invalid")
    path = Path(raw_path)
    if not path.is_absolute():
        path = MD_ROOT / path
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is not a regular file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected}, observed {actual}: {path}"
        )
    return path


def validate_p3_status(path: Path) -> dict[str, object]:
    """Validate the complete, immutable P3 approval basis for one target."""

    status = _load_json_object(path, "GROMOS P3 status")
    required_scalars = {
        "schema": "adman_gromos_repair_status/v3",
        "status": "VALIDATED_CONVERSION_AND_SMOKE_PASS",
        "molecule": "aDMan",
        "forcefield": "GROMOS 56A6@CARBO-R",
        "conversion_equivalence_validated": True,
        "official_native_total_force_validated": True,
        "gromacs_strict_em_nvt_smoke_passed": True,
        "eligible_to_count_as_repaired_forcefield_coverage": True,
        "production_ready": False,
        "production_md_permitted": False,
        "five_ns_run": False,
    }
    errors = [
        f"{key}={status.get(key)!r}, expected {expected!r}"
        for key, expected in required_scalars.items()
        if status.get(key) != expected
    ]
    gates = status.get("gates")
    if not isinstance(gates, Mapping):
        errors.append("gates object is missing")
    else:
        errors.extend(
            f"gate {key}={gates.get(key)!r}, expected {expected!r}"
            for key, expected in P3_REQUIRED_GATES.items()
            if gates.get(key) != expected
        )
    if errors:
        raise RuntimeError("GROMOS P3 gate failed: " + " | ".join(errors))

    equivalence_path = _verify_bound_file(status.get("equivalence"), "P3 equivalence")
    smoke_path = _verify_bound_file(status.get("smoke"), "P3 smoke")
    _verify_bound_file(status.get("source_provenance"), "P3 source provenance")
    _verify_bound_file(status.get("predeclared_tolerances"), "P3 tolerances")

    equivalence = _load_json_object(equivalence_path, "P3 equivalence")
    conformers = equivalence.get("conformers")
    if (
        equivalence.get("status") != "PASS_STRICT_ENERGY_FORCE_EQUIVALENCE"
        or not isinstance(conformers, list)
        or len(conformers) != 5
        or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("energy_kj_mol"), Mapping)
            or item["energy_kj_mol"].get("pass") is not True
            or not isinstance(item.get("force_kj_mol_nm"), Mapping)
            or item["force_kj_mol_nm"].get("pass") is not True
            for item in conformers
        )
    ):
        raise RuntimeError("P3 equivalence lacks five passing energy/force conformers")
    mapping = equivalence.get("atom_mapping")
    if not isinstance(mapping, Mapping) or mapping.get("status") != "PASS_IDENTITY_17_OF_17":
        raise RuntimeError("P3 equivalence lacks the identity 17-of-17 atom mapping")

    smoke = _load_json_object(smoke_path, "P3 smoke")
    strict = smoke.get("strict_preprocessing")
    log_checks = smoke.get("log_checks")
    nvt = smoke.get("nvt")
    if (
        smoke.get("status") != "PASS_DIAGNOSTIC_EM_NVT_SMOKE"
        or not isinstance(strict, Mapping)
        or strict.get("em_grompp_maxwarn") != 0
        or strict.get("nvt_grompp_maxwarn") != 0
        or strict.get("warning_count") != 0
        or not isinstance(log_checks, Mapping)
        or log_checks.get("em_no_fatal_error_warning_or_nan") is not True
        or log_checks.get("nvt_no_fatal_error_warning_or_nan") is not True
        or not isinstance(nvt, Mapping)
        or nvt.get("all_energy_values_finite") is not True
        or smoke.get("five_ns_or_production_run") is not False
    ):
        raise RuntimeError("P3 strict EM/NVT smoke evidence is incomplete")
    return status


def _resolve_approved_input(record: object, label: str) -> Path:
    return _verify_bound_file(record, f"approved GROMOS {label}")


def load_approved_entry(
    clean_name: str,
    *,
    manifest_path: Path = DEFAULT_APPROVED_MANIFEST,
    expected_molecule_name: str | None = None,
    expected_source_class: str | None = None,
) -> dict[str, object]:
    """Load one P3-bound production input and revalidate every current hash."""

    manifest = _load_json_object(manifest_path, "approved GROMOS manifest")
    if manifest.get("schema") != APPROVED_MANIFEST_SCHEMA:
        raise RuntimeError("unsupported approved GROMOS manifest schema")
    manifest_id = manifest.get("manifest_id")
    body = {key: value for key, value in manifest.items() if key != "manifest_id"}
    expected_id = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if manifest_id != expected_id:
        raise RuntimeError("approved GROMOS manifest_id does not match its contents")
    _verify_bound_file(manifest.get("generator"), "approved GROMOS manifest generator")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("approved GROMOS manifest entries are missing")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("clean_name") == clean_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{clean_name}: formal GROMOS production requires exactly one P3-approved "
            f"manifest entry; found {len(matches)}"
        )
    entry = matches[0]
    if expected_molecule_name is not None and entry.get("molecule_name") != expected_molecule_name:
        raise RuntimeError(f"{clean_name}: approved molecule identity mismatch")
    if expected_source_class is not None and entry.get("source_class") != expected_source_class:
        raise RuntimeError(f"{clean_name}: approved source-class mismatch")
    if entry.get("approval_basis") != "P3_STATUS_gates":
        raise RuntimeError(f"{clean_name}: approval is not bound to P3_STATUS gates")

    p3_path = _verify_bound_file(entry.get("p3_status"), f"{clean_name} P3 status")
    validate_p3_status(p3_path)
    package_path = _verify_bound_file(
        entry.get("package_manifest"), f"{clean_name} curated package manifest"
    )
    package = _load_json_object(package_path, "curated GROMOS package manifest")
    package_id = package.get("package_id")
    package_body = {key: value for key, value in package.items() if key != "package_id"}
    expected_package_id = hashlib.sha256(canonical_json_bytes(package_body)).hexdigest()
    package_record = entry.get("package_manifest")
    recorded_package_id = (
        package_record.get("package_id") if isinstance(package_record, Mapping) else None
    )
    if package_id != expected_package_id or recorded_package_id != package_id:
        raise RuntimeError(f"{clean_name}: curated package_id is invalid")
    inputs = entry.get("inputs")
    if not isinstance(inputs, Mapping):
        raise RuntimeError(f"{clean_name}: approved input table is missing")
    resolved = {
        label: _resolve_approved_input(inputs.get(label), label)
        for label in ("processed_gro", "topology", "metadata", "atom_map")
    }
    forcefield = inputs.get("forcefield")
    if not isinstance(forcefield, Mapping) or not isinstance(forcefield.get("path"), str):
        raise RuntimeError(f"{clean_name}: approved force-field tree record is missing")
    forcefield_path = Path(str(forcefield["path"]))
    if not forcefield_path.is_absolute():
        forcefield_path = MD_ROOT / forcefield_path
    if not forcefield_path.is_dir() or forcefield_path.is_symlink():
        raise RuntimeError(f"{clean_name}: approved force-field tree is invalid")
    recorded_tree = forcefield.get("tree")
    actual_tree = sha256_directory_tree(forcefield_path)
    if recorded_tree != actual_tree:
        raise RuntimeError(f"{clean_name}: approved force-field tree hash changed")
    package_files = package.get("files")
    package_forcefield = package.get("forcefield_tree")
    if not isinstance(package_files, Mapping) or any(
        package_files.get(label) != inputs.get(label)
        for label in ("processed_gro", "topology", "metadata", "atom_map")
    ):
        raise RuntimeError(f"{clean_name}: approval inputs differ from the curated package")
    if package_forcefield != forcefield:
        raise RuntimeError(f"{clean_name}: approval force-field differs from the curated package")

    metadata = _load_json_object(resolved["metadata"], "approved GROMOS metadata")
    validate_production_ready_claim(metadata)
    if not metadata_is_production_ready(metadata):
        raise RuntimeError(f"{clean_name}: curated metadata is not production ready")
    if metadata.get("p3_status_sha256") != sha256_file(p3_path):
        raise RuntimeError(f"{clean_name}: metadata is not bound to the current P3 status")
    _verify_bound_file(metadata.get("generator"), f"{clean_name} curated input generator")
    return {
        **entry,
        "resolved_inputs": {**resolved, "forcefield": forcefield_path},
        "approved_manifest_path": manifest_path,
        "approved_manifest_sha256": sha256_file(manifest_path),
        "approved_manifest_id": manifest_id,
    }
