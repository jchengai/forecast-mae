from typing import Any, Callable, Optional, Dict

import torch
from torchmetrics import Metric


class minADE(Metric):
    """Minimum Average Displacement Error
    minADE: The average L2 distance between the best forecasted trajectory and the ground truth.
            The best here refers to the trajectory that has the minimum endpoint error.
    """

    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(minADE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
        with torch.no_grad():
            pred = outputs["y_hat"]
            endpoint_error = torch.norm(
                pred[..., -1, :2] - target.unsqueeze(1)[..., -1, :2], p=2, dim=-1
            )
            best_mode = torch.argmin(endpoint_error, dim=-1)
            best_pred = pred[torch.arange(pred.shape[0]), best_mode]
            self.sum += torch.norm(best_pred - target, p=2, dim=-1).mean(-1).sum()
            self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
