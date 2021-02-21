import torch
from typing import Dict, List, Union
from pytorch_forecasting.metrics import MultiHorizonMetric, QuantileLoss
from utils.DILATE.dilate_loss import dilate_loss

class DirectionalQuantileLoss(MultiHorizonMetric):
    """
    Directional Quantile loss
    """
    def __init__(
        self,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        alpha: float = 0.01,
        **kwargs,
    ):
        super().__init__(quantiles=quantiles, **kwargs)
        self.alpha = alpha

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for i, q in enumerate(self.quantiles):
            # calculate quantile loss
            errors = target - y_pred[..., i]
            q_loss = torch.max((q - 1) * errors, q * errors)
            # calculate directional loss
            _x = y_pred[..., i].view(-1, 1).expand(-1, 2)
            _y = target.view(-1, 1).expand(-1, 2)
            pred_diff = (_x[:, 0] - _x[:, 1].view(-1, 1)).diag(1)
            target_diff = (_y[:, 0] - _y[:, 1].view(-1, 1)).diag(1)
            d_loss = self.alpha * abs(pred_diff - target_diff)
            # append sum of loss
            losses.append((q_loss + torch.cat((torch.Tensor([0.]).to(device=y_pred.device), d_loss), 0).view(-1, target.size(1))).unsqueeze(-1))
        losses = torch.cat(losses, dim=2)
        return losses

class DilateLoss(MultiHorizonMetric):
    """
    DILATE(DIstortion Loss with shApe and tImE)
    Vincent Le Guen et al. Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models. NeurIPS 2019. (https://arxiv.org/abs/1909.09020)
    """
    def __init__(
        self,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        alpha: float = 0.5,
        gamma: float = 0.01,
        **kwargs,
    ):
        super().__init__(quantiles=quantiles, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = y_pred.device
        total_losses = []

        # calculate DILATE
        for i, q in enumerate(self.quantiles):
            losses = []
            for j in range(y_pred.shape[0]):
                loss, loss_shape, loss_temporal = dilate_loss(y_pred[j,:,i].unsqueeze(0).unsqueeze(-1), target[j, :].unsqueeze(0).unsqueeze(-1), self.alpha, self.gamma, device)
                # append sum of loss
                loss = torch.repeat_interleave(loss.unsqueeze(-1).unsqueeze(-1), y_pred.shape[1], dim=0)
                losses.append(loss.unsqueeze(0))
            losses = torch.cat(losses, dim=0)
            total_losses.append(losses)
        total_losses = torch.cat(total_losses, dim=2)

        return total_losses


class DilateQuantileLoss(MultiHorizonMetric):
    """
    DILATE(DIstortion Loss with shApe and tImE) + Quantile Loss
    Vincent Le Guen et al. Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models. NeurIPS 2019. (https://arxiv.org/abs/1909.09020)
    """
    def __init__(
        self,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        alpha: float = 0.5,
        gamma: float = 0.01,
        weight: float = 0.01,
        **kwargs,
    ):
        super().__init__(quantiles=quantiles, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = y_pred.device

        # calculuate Quantile Loss
        quantile_losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., i]
            quantile_losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        total_quantile_losses = torch.cat(quantile_losses, dim=2)

        # calculate DILATE
        total_dilate_losses = []
        for i, q in enumerate(self.quantiles):
            dilate_losses = []
            for j in range(y_pred.shape[0]):
                loss, loss_shape, loss_temporal = dilate_loss(y_pred[j,:,i].unsqueeze(0).unsqueeze(-1), target[j, :].unsqueeze(0).unsqueeze(-1), self.alpha, self.gamma, device)
                # append sum of loss
                loss = torch.repeat_interleave(loss.unsqueeze(-1).unsqueeze(-1), y_pred.shape[1], dim=0)
                dilate_losses.append(loss.unsqueeze(0))
            dilate_losses = torch.cat(dilate_losses, dim=0)
            total_dilate_losses.append(dilate_losses)
        total_dilate_losses = torch.cat(total_dilate_losses, dim=2)

        losses = self.weight * total_quantile_losses + (1-self.weight) * total_dilate_losses

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