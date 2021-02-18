import torch
from typing import Dict, List, Union
from pytorch_forecasting.metrics import MultiHorizonMetric, QuantileLoss

class DirectionalQuantileLoss(MultiHorizonMetric):
    """
    Directional Quantile loss
    """
    def __init__(
        self,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        **kwargs,
    ):
        super().__init__(quantiles=quantiles, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        alpha = 1e-2
        for i, q in enumerate(self.quantiles):
            # calculate quantile loss
            errors = target - y_pred[..., i]
            q_loss = torch.max((q - 1) * errors, q * errors)
            # calculate directional loss
            _x = y_pred[..., i].view(-1, 1).expand(-1, 2)
            _y = target.view(-1, 1).expand(-1, 2)
            pred_diff = (_x[:, 0] - _x[:, 1].view(-1, 1)).diag(1)
            target_diff = (_y[:, 0] - _y[:, 1].view(-1, 1)).diag(1)
            d_loss = alpha * abs(pred_diff - target_diff)
            # append sum of loss
            losses.append((q_loss + torch.cat((torch.Tensor([0.]).to(device=y_pred.device), d_loss), 0).view(-1, target.size(1))).unsqueeze(-1))
        losses = torch.cat(losses, dim=2)
        return losses

def normalized_quantile_loss(actuals: torch.Tensor, predictions: torch.Tensor, quantiles: List[float] = None) -> torch.Tensor:
    """Computes normalized quantile loss for torch tensors.
    Uses the q-Risk metric as defined in the "Training Procedure" section of the
    main TFT paper.
    Args:
        actuals: Targets
        predictions: Predictions
        quantiles: Quantile list to use for loss calculations (between 0 & 1)
    Returns:
        Float tensor for normalized quantile loss.
    """
    normalizer = torch.sum(abs(actuals))
    if quantiles == None:
        QL = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    else:
        QL = QuantileLoss(quantiles=quantiles)

    q_loss = QL.loss(y_pred = predictions, target = actuals)
    reduced_q_loss = torch.sum(q_loss.reshape(-1, q_loss.shape[-1]), 0)
    normalized_loss = 2 * reduced_q_loss / normalizer
    return normalized_loss

def mean_directional_accuracy(actuals: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """ Mean Directional Accuracy """
    # actuals: torch.Size([756, 5])
    # predictions: torch.Size([756, 5, 3])
    actual = actuals
    predicted = predictions[:, :, 1]
    return torch.mean((torch.sign(actual[:, 1:] - actual[:, :-1]) == torch.sign(predicted[:, 1:] - predicted[:, :-1])).float())