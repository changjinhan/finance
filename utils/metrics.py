import torch
from pytorch_forecasting.metrics import QuantileLoss

def normalized_quantile_loss(actuals: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    normalizer = torch.sum(abs(actuals))
    QL = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    q_loss = QL.loss(y_pred = predictions, target = actuals)
    reduced_q_loss = torch.sum(q_loss.reshape(-1, q_loss.shape[-1]), 0)
    normalized_loss = 2 * reduced_q_loss / normalizer
    return normalized_loss