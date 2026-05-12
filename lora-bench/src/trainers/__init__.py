"""Custom trainers."""
from .op_dpo_gtw import (  # noqa: F401
    OnPolicyDPOGTWTrainer,
    StandaloneOnPolicyDPOGTW,
    OPDPOGTWConfig,
    cosine_beta,
    compute_group_loo_token_weights,
)
