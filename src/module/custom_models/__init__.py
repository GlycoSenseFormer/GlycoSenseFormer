from .baselines import (CompGCNNMR, EGNNNMR, GATNMR, GCNNMR, GINNMR, GeqShiftNMR, MPNNNMR, PaiNNNMR, RGCNNMR)
from .equiformer_glycanaa_cross_attention_fusion import (EquiformerV3GlycanAAIdentityResidualCrossAttentionFusionNMR, EquiformerV3GlycanAAIdentityResidualZeroMonoTokenCrossAttentionFusionNMR)
from .equiformer_v3_nmr import (EquiformerV3NMR)


def build_nmr_model(architecture="equiformer_v3", **kwargs):
    if architecture == "equiformer_v3":
        return EquiformerV3NMR(**kwargs)
    if architecture == "gcn":
        return GCNNMR(**kwargs)
    if architecture == "gin":
        return GINNMR(**kwargs)
    if architecture == "gat":
        return GATNMR(**kwargs)
    if architecture == "mpnn":
        return MPNNNMR(**kwargs)
    if architecture == "painn":
        return PaiNNNMR(**kwargs)
    if architecture == "egnn":
        return EGNNNMR(**kwargs)
    if architecture == "rgcn":
        return RGCNNMR(**kwargs)
    if architecture == "compgcn":
        return CompGCNNMR(**kwargs)
    if architecture == "geqshift":
        return GeqShiftNMR(**kwargs)
    if architecture == "eqv3_glycanaa_cross_attention_identity_residual_fusion":
        return EquiformerV3GlycanAAIdentityResidualCrossAttentionFusionNMR(**kwargs)
    if architecture == "eqv3_glycanaa_cross_attention_identity_residual_zero_mono_token_fusion":
        return EquiformerV3GlycanAAIdentityResidualZeroMonoTokenCrossAttentionFusionNMR(**kwargs)
    raise ValueError(f"Unknown publication model: {architecture}")
