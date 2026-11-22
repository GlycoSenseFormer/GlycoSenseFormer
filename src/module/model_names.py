"""Canonical name for the published model and its archived identifier."""

GLYCOSENSEFORMER = "GlycoSenseFormer"
LEGACY_GLYCOSENSEFORMER = "eqv3_glycanaa_cross_attention_identity_residual_fusion"


def canonical_model_name(name: str) -> str:
    if name in ("glycosenseformer", LEGACY_GLYCOSENSEFORMER):
        return GLYCOSENSEFORMER
    return name
