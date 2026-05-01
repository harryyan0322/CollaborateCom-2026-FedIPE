"""
Federated training loop for FedIPE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.federated_platforms import PlatformGraphData
from models.fedipe_model import FedIPEClientModel
from utils.viz_logger import VizLogger


@dataclass
class FedIPEConfig:
    rounds: int = 30
    local_epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    core_dim: int = 10
    hidden_dim: int = 64
    unified_dim: int = 128
    batch_size: int = 128
    temperature: float = 0.1
    alpha_start: float = 0.1
    alpha_end: float = 0.01
    anchor_weight: float = 0.5
    subgraph_sample_size: int = 48
    seed: int = 42


class FedIPETrainer:
    def __init__(
        self,
        platforms: List[PlatformGraphData],
        config: FedIPEConfig,
        device: str = "cpu",
    ) -> None:
        self.platforms = platforms
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.clients = {
            platform.platform_id: FedIPEClientModel(
                explicit_input_dim=platform.explicit_local.size(1),
                node_feature_dim=platform.node_features.size(1),
                core_dim=config.core_dim,
                hidden_dim=config.hidden_dim,
                unified_dim=config.unified_dim,
            ).to(self.device)
            for platform in platforms
        }
        first_client = next(iter(self.clients.values()))
        self.global_shared_state = first_client.shared_state_dict()
        self.history: List[Dict[str, float]] = []

    def _alpha_for_round(self, round_idx: int) -> float:
        if self.config.rounds <= 1:
            return self.config.alpha_end
        progress = round_idx / max(1, self.config.rounds - 1)
        return self.config.alpha_start + (self.config.alpha_end - self.config.alpha_start) * progress

    def _local_update(
        self,
        platform: PlatformGraphData,
        model: FedIPEClientModel,
        alpha: float,
    ) -> tuple[Dict[str, torch.Tensor], int, Dict[str, float]]:
        model.load_shared_state_dict(self.global_shared_state)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        train_idx = platform.train_indices.to(self.device)
        labels = platform.labels.to(self.device)
        last_pred_loss = 0.0
        last_anchor_loss = 0.0
        last_ctr_loss = 0.0
        for _ in range(self.config.local_epochs):
            optimizer.zero_grad()
            output = model(
                platform,
                temperature=self.config.temperature,
                subgraph_sample_size=self.config.subgraph_sample_size,
            )
            pred_loss = torch.nn.functional.mse_loss(output.predictions.index_select(0, train_idx), labels.index_select(0, train_idx))
            loss = pred_loss + self.config.anchor_weight * output.anchor_loss + alpha * output.contrastive_loss
            loss.backward()
            optimizer.step()
            last_pred_loss = float(pred_loss.detach().cpu())
            last_anchor_loss = float(output.anchor_loss.detach().cpu())
            last_ctr_loss = float(output.contrastive_loss.detach().cpu())
        return (
            model.shared_state_dict(),
            max(1, int(train_idx.numel())),
            {
                "pred_loss": last_pred_loss,
                "anchor_loss": last_anchor_loss,
                "contrastive_loss": last_ctr_loss,
            },
        )

    def _aggregate_shared_states(
        self,
        local_states: List[Dict[str, torch.Tensor]],
        weights: List[int],
    ) -> Dict[str, torch.Tensor]:
        total_weight = float(sum(weights))
        aggregated: Dict[str, torch.Tensor] = {}
        for key in local_states[0].keys():
            acc = None
            for state, weight in zip(local_states, weights):
                tensor = state[key].float() * (weight / total_weight)
                acc = tensor if acc is None else acc + tensor
            aggregated[key] = acc
        return aggregated

    def _evaluate_platform(self, platform: PlatformGraphData, model: FedIPEClientModel) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            output = model(
                platform,
                temperature=self.config.temperature,
                subgraph_sample_size=self.config.subgraph_sample_size,
            )
        test_idx = platform.test_indices.to(self.device)
        y_true = platform.labels.index_select(0, platform.test_indices).cpu().numpy()
        y_pred = output.predictions.index_select(0, test_idx).detach().cpu().numpy()
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        try:
            r2 = float(r2_score(y_true, y_pred))
        except Exception:
            r2 = 0.0
        model.train()
        return {
            "platform_id": platform.platform_id,
            "platform_name": platform.display_name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "num_test": int(len(y_true)),
        }

    def _alignment_score(self) -> float:
        banks = []
        for platform in self.platforms:
            model = self.clients[platform.platform_id]
            model.eval()
            with torch.no_grad():
                output = model(
                    platform,
                    temperature=self.config.temperature,
                    subgraph_sample_size=self.config.subgraph_sample_size,
                )
            model.train()
            if output.subgraph_embeddings.size(0) == 0:
                continue
            banks.append(
                (
                    output.subgraph_embeddings.detach().cpu().numpy(),
                    output.structural_attributes.detach().cpu().numpy(),
                )
            )
        if len(banks) < 2:
            return 0.0
        distances = []
        for idx in range(len(banks)):
            for jdx in range(idx + 1, len(banks)):
                emb_a, attr_a = banks[idx]
                emb_b, attr_b = banks[jdx]
                attr_a_norm = (attr_a - attr_a.mean(axis=0, keepdims=True)) / (attr_a.std(axis=0, keepdims=True) + 1e-6)
                attr_b_norm = (attr_b - attr_b.mean(axis=0, keepdims=True)) / (attr_b.std(axis=0, keepdims=True) + 1e-6)
                attr_dist = ((attr_a_norm[:, None, :] - attr_b_norm[None, :, :]) ** 2).sum(axis=2)
                match_idx = attr_dist.argmin(axis=1)
                paired = np.linalg.norm(emb_a - emb_b[match_idx], axis=1)
                distances.extend(paired.tolist())
        return float(np.mean(distances)) if distances else 0.0

    def train(self) -> Dict[str, object]:
        for round_idx in range(self.config.rounds):
            alpha = self._alpha_for_round(round_idx)
            shared_states = []
            weights = []
            round_logs = []
            for platform in self.platforms:
                client = self.clients[platform.platform_id]
                state, weight, log_row = self._local_update(platform, client, alpha=alpha)
                shared_states.append(state)
                weights.append(weight)
                round_logs.append(log_row)
            self.global_shared_state = self._aggregate_shared_states(shared_states, weights)
            for client in self.clients.values():
                client.load_shared_state_dict(self.global_shared_state)

            eval_rows = [self._evaluate_platform(platform, self.clients[platform.platform_id]) for platform in self.platforms]
            avg_rmse = float(np.mean([row["RMSE"] for row in eval_rows]))
            avg_mae = float(np.mean([row["MAE"] for row in eval_rows]))
            avg_r2 = float(np.mean([row["R2"] for row in eval_rows]))
            self.history.append(
                {
                    "round": round_idx + 1,
                    "alpha": alpha,
                    "avg_rmse": avg_rmse,
                    "avg_mae": avg_mae,
                    "avg_r2": avg_r2,
                    "avg_pred_loss": float(np.mean([row["pred_loss"] for row in round_logs])),
                    "avg_anchor_loss": float(np.mean([row["anchor_loss"] for row in round_logs])),
                    "avg_contrastive_loss": float(np.mean([row["contrastive_loss"] for row in round_logs])),
                }
            )

        final_rows = [self._evaluate_platform(platform, self.clients[platform.platform_id]) for platform in self.platforms]
        summary = {
            "config": asdict(self.config),
            "average_metrics": {
                "RMSE": float(np.mean([row["RMSE"] for row in final_rows])),
                "MAE": float(np.mean([row["MAE"] for row in final_rows])),
                "R2": float(np.mean([row["R2"] for row in final_rows])),
                "alignment_score": self._alignment_score(),
            },
            "platform_metrics": final_rows,
            "history": self.history,
            "platform_metadata": {
                platform.platform_id: platform.metadata for platform in self.platforms
            },
        }
        return summary

    def save_outputs(self, summary: Dict[str, object], result_dir: str | Path = "results") -> None:
        result_root = Path(result_dir)
        result_root.mkdir(parents=True, exist_ok=True)
        viz = VizLogger(base_dir=result_root)
        viz.save_json(summary, "fedipe_summary.json")
        viz.save_csv(summary["platform_metrics"], "fedipe_platform_metrics.csv")
        viz.save_csv(summary["history"], "fedipe_training_history.csv")
        (result_root / "fedipe_config.json").write_text(
            json.dumps(summary["config"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


__all__ = ["FedIPEConfig", "FedIPETrainer"]
