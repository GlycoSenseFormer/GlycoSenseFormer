#!/usr/bin/env python3
"""Strict metadata policy shared by CHARMM topology gates and runners."""

from __future__ import annotations

import math
from typing import Any


DEFAULT_CHARGE_TOLERANCE = 0.001
APPROVED_SCHEMA_VERSION = 2
PATCH_EXPANSION_POLICY = "charmm36_standard_patch_expansion_v1"
ALIGNMENT364_PATCH_EXPANSION_POLICY = (
    "charmm36_standard_alignment364_patch_expansion_v1"
)
URONIC_RESIDUE_POLICY = "charmm36_standard_uronic_residue_v1"
APPROVED_TOPOLOGY_POLICIES = {
    PATCH_EXPANSION_POLICY,
    ALIGNMENT364_PATCH_EXPANSION_POLICY,
    URONIC_RESIDUE_POLICY,
}
APPROVED_PATCH_MANIFEST_SHA256 = "1a37d6e2bada274948f43ded9ace4d431e5c060148622dffda3e9e5fee38fbec"
APPROVED_ALIGNMENT364_PATCH_MANIFEST_SHA256 = (
    "20f29fe96ab93f7faa22258938e185ec1c02149f98a4d75c0ea5ab4c92423b94"
)
APPROVED_SOURCE_FF_TREE_SHA256 = "9d942a07f4cbb7c70f8314b6a8f5108bbe9a2f49b4a463477c18acc2656ee92e"


def assess_charmm_standard_metadata(
    metadata: object,
    *,
    charge_tolerance: float = DEFAULT_CHARGE_TOLERANCE,
    expected_molecule_name: str | None = None,
    expected_clean_name: str | None = None,
    expected_source_class: str | None = None,
) -> dict[str, Any]:
    """Validate production approval and charge/neutralization semantics.

    Neutral standard-patch solutes and charged standard residues are both
    supported, but their metadata must state the matching policy explicitly.
    This function deliberately does not infer approval from filenames.
    """

    errors: list[str] = []
    result: dict[str, Any] = {
        "approved": False,
        "errors": errors,
        "charge_e": None,
        "requires_neutralization": None,
    }
    if not isinstance(metadata, dict):
        errors.append("metadata root is not an object")
        return result
    if metadata.get("production_ready") is not True:
        errors.append("production_ready is not explicitly true")
    if metadata.get("production_ready_scope") != "solute topology only":
        errors.append("production_ready_scope is not 'solute topology only'")
    if metadata.get("schema_version") != APPROVED_SCHEMA_VERSION:
        errors.append(
            f"schema_version is not the approved value {APPROVED_SCHEMA_VERSION}"
        )
    topology_policy = metadata.get("topology_policy")
    if topology_policy not in APPROVED_TOPOLOGY_POLICIES:
        errors.append(f"unsupported or missing topology_policy: {topology_policy!r}")
    if not isinstance(metadata.get("generator"), dict):
        errors.append("generator provenance block is missing")

    if topology_policy in {
        PATCH_EXPANSION_POLICY,
        ALIGNMENT364_PATCH_EXPANSION_POLICY,
    }:
        forcefield = metadata.get("forcefield")
        if not isinstance(forcefield, dict):
            errors.append("standard-patch metadata lacks forcefield provenance")
        else:
            approved_manifest = (
                APPROVED_PATCH_MANIFEST_SHA256
                if topology_policy == PATCH_EXPANSION_POLICY
                else APPROVED_ALIGNMENT364_PATCH_MANIFEST_SHA256
            )
            if forcefield.get("manifest_sha256") != approved_manifest:
                errors.append("standard-patch manifest SHA256 is not approved")
            source_tree = forcefield.get("source_forcefield_tree")
            if not isinstance(source_tree, dict) or source_tree.get("sha256") != APPROVED_SOURCE_FF_TREE_SHA256:
                errors.append("standard-patch source force-field tree SHA256 is not approved")

    expected_identity = {
        "molecule_name": expected_molecule_name,
        "clean_name": expected_clean_name,
        "source_class": expected_source_class,
    }
    for field, expected_value in expected_identity.items():
        if expected_value is not None and metadata.get(field) != expected_value:
            errors.append(
                f"{field} does not match the queued target: "
                f"{metadata.get(field)!r} vs {expected_value!r}"
            )

    observed = metadata.get("observed_topology_charge_e")
    expected = metadata.get("expected_solute_charge_e")
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        errors.append("observed_topology_charge_e is not numeric")
        observed_value = None
    else:
        observed_value = float(observed)
        if not math.isfinite(observed_value):
            errors.append("observed_topology_charge_e is not finite")
            observed_value = None
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        errors.append("expected_solute_charge_e is not numeric")
        expected_value = None
    else:
        expected_value = float(expected)
        if not math.isfinite(expected_value):
            errors.append("expected_solute_charge_e is not finite")
            expected_value = None

    if observed_value is not None:
        result["charge_e"] = observed_value
        if abs(observed_value - round(observed_value)) > charge_tolerance:
            errors.append(f"observed topology charge is non-integer: {observed_value:+.6f} e")
    if observed_value is not None and expected_value is not None:
        if abs(observed_value - expected_value) > charge_tolerance:
            errors.append(
                "expected_solute_charge_e disagrees with observed_topology_charge_e: "
                f"{expected_value:+.6f} vs {observed_value:+.6f} e"
            )

    requires = metadata.get("requires_neutralization")
    if type(requires) is not bool:
        errors.append("requires_neutralization is not an explicit boolean")
        requires_value = None
    else:
        requires_value = requires
        result["requires_neutralization"] = requires_value

    if observed_value is not None and requires_value is not None:
        is_charged = abs(observed_value) > charge_tolerance
        if is_charged != requires_value:
            errors.append(
                "requires_neutralization does not match the observed topology charge"
            )

        counterions = metadata.get("required_counterions")
        if is_charged:
            if not isinstance(counterions, dict):
                errors.append("charged topology lacks required_counterions metadata")
            else:
                count = counterions.get("count")
                ion_charge = counterions.get("charge_e")
                expected_count = abs(int(round(observed_value)))
                expected_ion_charge = -1 if observed_value > 0 else 1
                if count != expected_count:
                    errors.append(
                        f"required_counterions.count is {count!r}, expected {expected_count}"
                    )
                if ion_charge != expected_ion_charge:
                    errors.append(
                        "required_counterions.charge_e has the wrong sign/value: "
                        f"{ion_charge!r}, expected {expected_ion_charge}"
                    )
        elif isinstance(counterions, dict) and counterions.get("count") not in (None, 0):
            errors.append("neutral topology requests a non-zero counterion count")

    result["approved"] = not errors
    return result
