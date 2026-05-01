"""
FedIPE main entrypoint.

The repository now follows the revised paper setting:
1. Construct three heterogeneous platforms (A/B/C)
2. Train private explicit adapters + shared explicit projector
3. Train private low-level GNN layers + shared high-level GNN layers
4. Fuse explicit/implicit embeddings for propagation heat prediction
5. Aggregate only shared parameters with FedAvg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.federated_platforms import FederatedPlatformBuilder
from system.fedipe_trainer import FedIPEConfig, FedIPETrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_real", action="store_true", help="Use the preprocessed real corpus.")
    parser.add_argument("--device", type=str, default="cpu", help="Training device.")
    parser.add_argument("--rounds", type=int, default=30, help="Federated communication rounds.")
    parser.add_argument("--local_epochs", type=int, default=3, help="Local epochs per round.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--users_per_platform", type=int, default=256, help="Maximum labeled users per platform.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Contrastive temperature.")
    parser.add_argument("--alpha_start", type=float, default=0.1, help="Initial contrastive weight.")
    parser.add_argument("--alpha_end", type=float, default=0.01, help="Final contrastive weight.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    print("=" * 72)
    print("FedIPE: Federated Heterogeneous Feature Alignment for Propagation Evaluation")
    print("=" * 72)

    builder = FederatedPlatformBuilder(
        max_users_per_platform=args.users_per_platform,
        seed=args.seed,
    )
    platforms = builder.build(use_real=args.use_real)

    print("\n[Platform Construction]")
    for platform in platforms:
        print(
            f"- {platform.display_name}: "
            f"{platform.metadata['num_users']} target users / "
            f"{platform.metadata['num_posts']} posts / "
            f"{platform.metadata['num_edges']} edges / "
            f"explicit dim {platform.metadata['explicit_dim']} / "
            f"topology={platform.metadata['topology']}"
        )

    config = FedIPEConfig(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        temperature=args.temperature,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        seed=args.seed,
    )
    trainer = FedIPETrainer(platforms=platforms, config=config, device=args.device)
    summary = trainer.train()
    trainer.save_outputs(summary, result_dir=Path("results"))

    avg = summary["average_metrics"]
    print("\n[Final Metrics]")
    print(f"Average RMSE: {avg['RMSE']:.4f}")
    print(f"Average MAE : {avg['MAE']:.4f}")
    print(f"Average R2  : {avg['R2']:.4f}")
    print(f"Alignment   : {avg['alignment_score']:.4f}")
    print("\nSaved to results/fedipe_summary.json")

    return summary


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["average_metrics"], ensure_ascii=False, indent=2))
