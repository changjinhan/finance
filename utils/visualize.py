import os
import numpy as np
import pandas as pd
import torch

from utils.metrics import normalized_quantile_loss
from utils.models import SparseTemporalFusionTransformer

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.metrics import QuantileLoss, SMAPE

def visualize(dataset, dataloader, best_model, image_root) -> None:
    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    actuals = torch.cat([y[0] for x, y in iter(dataloader)])
    raw_predictions, x = best_model.predict(dataloader, mode="raw", return_x=True)
    normalizer = torch.sum(abs(actuals))
    q_losses = QuantileLoss([0.1,0.5,0.9]).loss(raw_predictions["prediction"], actuals)
    mean_losses = 2*torch.sum(q_losses.reshape(q_losses.shape[0], -1), 1) / normalizer
    indices = torch.flip(mean_losses.argsort(descending=True), (0,))  # sort losses

    for idx in range(len(raw_predictions['groups'])): 
        group_name = []
        try:
            group_ids = dataset.group_ids
            embedding_labels = best_model.hparams.embedding_labels
            for i, emb in enumerate(raw_predictions['groups'][indices[idx]]):
                for name, val in embedding_labels[group_ids[i]].items():
                    if val == emb:
                        group_name.append(name)
                    
            fig = best_model.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=True, group_name=group_name)
            fig.savefig(os.path.join(image_root, f'best_QLOSS_{idx}_{group_name}.png'))
        except:
            continue

    # prediction plot sort by SMAPE 
    predictions = best_model.predict(dataloader)
    mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
    indices = torch.flip(mean_losses.argsort(descending=True), (0,))  # sort losses

    for idx in range(len(raw_predictions['groups'])): 
        group_name = []
        try:
            group_ids = dataset.group_ids
            embedding_labels = best_model.hparams.embedding_labels
            for i, emb in enumerate(raw_predictions['groups'][indices[idx]]):
                for name, val in embedding_labels[group_ids[i]].items():
                    if val == emb:
                        group_name.append(name)

            fig2 = best_model.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=SMAPE(), group_name=group_name)
            fig2.savefig(os.path.join(image_root, f'best_SMAPE_{idx}_{group_name}.png'))
        except:
            continue

    predictions, x = best_model.predict(dataloader, return_x=True)
    predictions_vs_actuals = best_model.calculate_prediction_actual_by_variable(x, predictions)
    fig3_dict = best_model.plot_prediction_actual_by_variable(predictions_vs_actuals)
    for name in fig3_dict.keys():
        fig3_dict[name].savefig(os.path.join(image_root, f'prediction_vs_actuals_{name}.png'))

    interpretation = best_model.interpret_output(raw_predictions, reduction="sum")
    fig4_dict = best_model.plot_interpretation(interpretation)
    for name in fig4_dict.keys():
        fig4_dict[name].savefig(os.path.join(image_root, f'interpretation_{name}.png'))