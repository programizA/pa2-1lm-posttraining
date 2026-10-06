from __future__ import annotations

import torch

from common.metrics import masked_mean, sampled_kl, sample_entropy


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6):
    """Return one scalar advantage per sampled completion.

    `group_ids[i]` identifies which prompt produced reward `rewards[i]`.
    Validate this implementation against the group-relative definition in the assignment manual.
    """
    # FIX (deliberate starter defect): the starter normalised with the mean/std of the WHOLE batch,
    # ignoring group_ids. GRPO's baseline is the mean of the K completions for the SAME prompt,
    # A_k = (r_k - mean_group) / (std_group + eps). With a global baseline an easy prompt's
    # completions all receive positive advantage and a hard prompt's all receive negative
    # advantage, so the update learns "which prompt was easy" instead of "which answer was better",
    # and a zero-variance group still produces a non-zero signal.
    rewards = rewards.float()
    group_ids = torch.as_tensor(group_ids, device=rewards.device)
    adv = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        idx = group_ids == g
        r = rewards[idx]
        mean = r.mean()
        std = r.std(unbiased=False)
        adv[idx] = (r - mean) / (std + eps)
    return adv


def group_statistics(rewards: torch.Tensor, group_ids: torch.Tensor, tol: float = 1e-6):
    """Per-group reward std and informative-group flags (std > tol)."""
    rewards = rewards.float()
    group_ids = torch.as_tensor(group_ids, device=rewards.device)
    stds = []
    for g in torch.unique(group_ids):
        stds.append(float(rewards[group_ids == g].std(unbiased=False).item()))
    stds_t = torch.tensor(stds)
    informative = stds_t > tol
    return {
        "group_reward_std_mean": float(stds_t.mean().item()) if len(stds) else 0.0,
        "uninformative_group_fraction": float((~informative).float().mean().item()) if len(stds) else 0.0,
        "informative_group_fraction": float(informative.float().mean().item()) if len(stds) else 0.0,
        "group_reward_stds": stds,
    }


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    `token_mask` may be all-zero for a completion that was deliberately masked because it hit the
    maximum generation length.
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask).sum(-1)
    if loss_type == "grpo":
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        # Constant normalization rather than dividing by each response's realized length.
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def grpo_sequence_loss(
    new_logp_i,
    old_logp_i,
    adv_i,
    mask_i,
    ref_logp_i,
    eps,
    beta,
    n_sequences: int,
    total_valid_tokens: float,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """Contribution of ONE completion to grpo_policy_loss (summing over completions gives it back).

    Lets the training loop back-propagate one completion at a time, which (i) bounds activation
    memory and (ii) yields each completion's own gradient norm, the length-conditioned statistic
    used for the canonical-vs-Dr.GRPO normalisation comparison. Shapes: [1, T] and [1].
    """
    ratio = torch.exp(new_logp_i - old_logp_i)
    adv = adv_i[:, None]
    obj = torch.minimum(ratio * adv, ratio.clamp(1.0 - eps, 1.0 + eps) * adv)
    token_sum = (obj * mask_i).sum(-1)
    if loss_type == "grpo":
        per_seq = token_sum / mask_i.sum(-1).clamp_min(1.0)
    elif loss_type == "dr_grpo":
        per_seq = token_sum / float(max_completion_length)
    else:
        raise ValueError(loss_type)
    policy_part = -per_seq.sum() / float(n_sequences)
    lr = ref_logp_i - new_logp_i
    kl_tok = torch.exp(lr) - lr - 1.0
    kl_part = (kl_tok * mask_i).sum() / max(float(total_valid_tokens), 1.0)
    return policy_part + float(beta) * kl_part, {"policy_part": policy_part, "kl_part": kl_part}


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
