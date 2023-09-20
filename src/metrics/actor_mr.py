from typing import Any, Callable, Dict, Optional

import torch
from torchmetrics import Metric


class ActorMR(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        miss_threshold: float = 2.0,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(ActorMR, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.miss_threshold = miss_threshold

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        scored_mask: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            y_hat = outputs["y_hat"]
            bs, K, N, T, _ = y_hat.shape
            valid_mask = scored_mask.unsqueeze(1).float()  # [B, 1, N]
            num_scored_agents = valid_mask.long().sum(-1)
            fde = torch.norm(
                y_hat[..., -1, :2] - target.unsqueeze(1)[..., -1, :2], dim=-1
            )
            avg_fde = (fde * valid_mask).sum(-1) / num_scored_agents

            best_world = torch.argmin(avg_fde, dim=-1)
            best_world_fde = fde[torch.arange(bs), best_world]
            missed_predictions = best_world_fde > self.miss_threshold  # [b, N]
            missed_predictions[~scored_mask] = False

            self.sum += missed_predictions.sum()
            self.count += num_scored_agents.sum()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
