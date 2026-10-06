from __future__ import annotations

import argparse

from common.data import load_yaml
from common.evaluation import evaluate_rl_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the untouched SFT policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-prompts", type=int)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    evaluate_rl_policy(cfg, args.adapter, args.name, cfg["results_dir"], args.max_prompts, args.greedy)


if __name__ == "__main__":
    main()
