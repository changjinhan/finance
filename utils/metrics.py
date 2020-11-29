import torch
from typing import Dict, List, Union
from pytorch_forecasting.metrics import QuantileLoss

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