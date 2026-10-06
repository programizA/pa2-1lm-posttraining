"""Unit tests that validate the three corrected objectives against the manual's equations.

Run: python -m pytest -q tests/test_objectives.py   (or: python -m tests.test_objectives)
"""
from __future__ import annotations

import math

import torch

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import clip_diagnostics, compute_gae, ppo_policy_loss, shaped_rewards
from task3_grpo.grpo import group_relative_advantages, group_statistics, grpo_policy_loss


# ----------------------------------------------------------------------------- DPO
def test_dpo_at_initialisation_is_log2():
    # policy == reference  ->  m = 0  ->  loss = -log sigmoid(0) = log 2, for ANY reference margin.
    ref_c = torch.tensor([-10.0, -3.0, -50.0])
    ref_r = torch.tensor([-2.0, -9.0, -20.0])
    loss, diag = dpo_loss(ref_c.clone(), ref_r.clone(), ref_c, ref_r, beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6
    assert abs(diag["margin_mean"].item()) < 1e-6
    assert diag["preference_accuracy"].item() == 0.0  # strict m > 0


def test_dpo_matches_hand_formula():
    pc, pr = torch.tensor([-5.0]), torch.tensor([-7.0])
    rc, rr = torch.tensor([-6.0]), torch.tensor([-6.5])
    beta = 0.3
    m = (pc - rc) - (pr - rr)  # (1.0) - (-0.5) = 1.5
    expected = -torch.nn.functional.logsigmoid(beta * m)
    loss, diag = dpo_loss(pc, pr, rc, rr, beta)
    assert torch.allclose(loss, expected.mean())
    assert abs(diag["margin_mean"].item() - 1.5) < 1e-6
    assert diag["preference_accuracy"].item() == 1.0


def test_dpo_gradient_direction():
    # Gradient descent must raise chosen log-prob and lower rejected log-prob.
    pc = torch.tensor([-5.0], requires_grad=True)
    pr = torch.tensor([-5.0], requires_grad=True)
    loss, _ = dpo_loss(pc, pr, torch.tensor([-5.0]), torch.tensor([-5.0]), 0.1)
    loss.backward()
    assert pc.grad.item() < 0 and pr.grad.item() > 0


# ----------------------------------------------------------------------------- PPO
def _grad_of_loss(new_lp, old_lp, adv, eps=0.2):
    new_lp = new_lp.clone().requires_grad_(True)
    mask = torch.ones_like(new_lp)
    loss, _, _ = ppo_policy_loss(new_lp, old_lp, adv, mask, eps)
    loss.backward()
    return new_lp.grad


def test_ppo_clipping_geometry():
    eps = 0.2
    old = torch.zeros(1, 4)
    # ratios: 1.5 (A>0, should be clipped -> 0 grad), 1.5 (A<0, NOT clipped -> grad),
    #         0.5 (A<0, clipped -> 0 grad), 0.5 (A>0, NOT clipped -> grad)
    new = torch.log(torch.tensor([[1.5, 1.5, 0.5, 0.5]]))
    adv = torch.tensor([[1.0, -1.0, -1.0, 1.0]])
    g = _grad_of_loss(new, old, adv, eps)[0]
    assert g[0].item() == 0.0 and g[2].item() == 0.0
    assert g[1].item() != 0.0 and g[3].item() != 0.0
    d = clip_diagnostics(torch.exp(new - old), adv, torch.ones_like(adv), eps)
    assert d["clip_fraction"] == 1.0
    assert d["affected_fraction"] == 0.5


def test_ppo_objective_is_min():
    eps = 0.2
    old = torch.zeros(1, 1)
    new = torch.log(torch.tensor([[2.0]]))
    adv = torch.tensor([[1.0]])
    loss, _, _ = ppo_policy_loss(new, old, adv, torch.ones_like(adv), eps)
    assert abs(loss.item() - (-(1.0 + eps))) < 1e-6  # min(2*1, 1.2*1) = 1.2


def test_ppo_no_clip_at_ratio_one():
    old = torch.randn(2, 5)
    adv = torch.randn(2, 5)
    loss, ratio, cf = ppo_policy_loss(old.clone(), old, adv, torch.ones_like(adv), 0.2)
    assert cf.item() == 0.0
    assert torch.allclose(loss, -adv.mean())


def test_gae_terminal_reward_lambda1():
    # gamma=lambda=1, V=0 -> every advantage equals the terminal reward; padding stays 0.
    rewards = torch.tensor([[0.0, 0.0, 2.0, 0.0]])
    values = torch.zeros_like(rewards)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.tensor([[2.0, 2.0, 2.0, 0.0]]))


def test_shaped_rewards_terminal_position():
    task = torch.tensor([3.0])
    lp = torch.tensor([[-1.0, -1.0, -1.0, 0.0]])
    ref = torch.tensor([[-1.5, -1.0, -0.5, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    r = shaped_rewards(task, lp, ref, mask, beta_kl=0.1)
    assert torch.allclose(r, torch.tensor([[-0.05, 0.0, 3.05, 0.0]]))


# ----------------------------------------------------------------------------- GRPO
def test_grpo_advantages_are_per_group():
    # Group 0 is an "easy" prompt (high rewards), group 1 a "hard" one (low rewards).
    rewards = torch.tensor([9.0, 11.0, -11.0, -9.0])
    gids = torch.tensor([0, 0, 1, 1])
    adv = group_relative_advantages(rewards, gids)
    # Within each group the better completion gets +1, the worse -1, regardless of prompt difficulty.
    assert torch.allclose(adv, torch.tensor([-1.0, 1.0, -1.0, 1.0]), atol=1e-4)


def test_grpo_zero_variance_group_gives_zero_advantage():
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    gids = torch.tensor([0, 0, 0, 1, 1, 1])
    adv = group_relative_advantages(rewards, gids)
    assert torch.allclose(adv[:3], torch.zeros(3))
    st = group_statistics(rewards, gids)
    assert st["uninformative_group_fraction"] == 0.5


def test_grpo_loss_normalisation():
    # Two sequences of length 2 and 4 with identical ratio 1 and advantage 1.
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    lp = torch.zeros(2, 4)
    adv = torch.tensor([1.0, 1.0])
    l_grpo, _ = grpo_policy_loss(lp, lp, adv, mask, lp, 0.2, 0.0, "grpo")
    l_dr, _ = grpo_policy_loss(lp, lp, adv, mask, lp, 0.2, 0.0, "dr_grpo", max_completion_length=4)
    assert abs(l_grpo.item() - (-1.0)) < 1e-6           # each seq contributes 1
    assert abs(l_dr.item() - (-(2 / 4 + 4 / 4) / 2)) < 1e-6  # contributions scale with length


def test_grpo_sequence_decomposition_sums_to_batch_loss():
    from task3_grpo.grpo import grpo_sequence_loss
    torch.manual_seed(0)
    B, T = 4, 6
    new, old, ref = torch.randn(B, T) * 0.1 - 2, torch.randn(B, T) * 0.1 - 2, torch.randn(B, T) * 0.1 - 2
    mask = (torch.arange(T)[None, :] < torch.tensor([[6], [3], [0], [5]])).float()  # incl. a masked completion
    adv = torch.randn(B)
    for lt in ("grpo", "dr_grpo"):
        full, _ = grpo_policy_loss(new, old, adv, mask, ref, 0.2, 0.1, lt, max_completion_length=T)
        parts = sum(grpo_sequence_loss(new[i:i+1], old[i:i+1], adv[i:i+1], mask[i:i+1], ref[i:i+1], 0.2, 0.1,
                                       B, float(mask.sum()), lt, T)[0] for i in range(B))
        assert torch.allclose(full, parts, atol=1e-6), (lt, full, parts)


if __name__ == "__main__":
    import sys
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")
    sys.exit(0)
