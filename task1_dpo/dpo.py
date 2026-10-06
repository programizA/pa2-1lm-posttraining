from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    L = -E[ log sigmoid( beta * ( [log pi(y+|x) - log ref(y+|x)] - [log pi(y-|x) - log ref(y-|x)] ) ) ]

    All inputs are per-example *sequence* log-probabilities (summed over response tokens).

    FIX (deliberate starter defect): the starter computed beta * (policy_margin + ref_margin).
    Rearranging the DPO objective gives beta * (policy_margin - ref_margin), i.e. the
    reference margin is SUBTRACTED. With "+", at initialisation (policy == ref) the logits are
    2*beta*ref_margin instead of 0, so the loss is not log(2) and the gradient rewards
    whatever preference the reference already had rather than the change relative to it.
    """

    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp

    # m_theta in the manual: implicit-reward margin of the chosen over the rejected response.
    margin = policy_margin - ref_margin
    logits = beta * margin

    loss = -F.logsigmoid(logits).mean()

    chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)

    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "margin_mean": margin.detach().mean(),
        "chosen_reward_mean": chosen_reward.detach().mean(),
        "rejected_reward_mean": rejected_reward.detach().mean(),
        "preference_accuracy": (margin > 0).float().mean().detach(),
    }
