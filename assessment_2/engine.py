import gc
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score


class CustomTrainer(Trainer):
    def __init__(self, data_type="tile", **kwargs):
        super().__init__(**kwargs)
        self.data_type = data_type
        self.loss_fn = nn.BCELoss()

    def compute_loss(self, model: any, inputs: Dict, return_outputs=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        outputs = model(
            inputs["layout_feat"],
            inputs["role_feat"],
            inputs["edge_index"],
        )
        loss = self.loss_fn(outputs, inputs["target"].to(device))

        if return_outputs:
            return (loss, outputs)
        return loss

    def create_optimizer(self):
        model = self.model
        no_decay = []
        for n, m in model.named_modules():
            if isinstance(
                m,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.LayerNorm,
                    torch.nn.LayerNorm,
                    torch.nn.GroupNorm,
                ),
            ):
                no_decay.append(n)

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        del inputs["layout_feat"]
        del inputs["role_feat"]
        del inputs["edge_index"]

        gc.collect()

        predictions = outputs.cpu().detach()
        return loss, predictions, inputs["target"].cpu().detach()


def compute_metrics_fn(eval_preds):
    predictions, labels = eval_preds

    predictions = torch.sigmoid(predictions)
    predictions = predictions.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    
    predictions = np.where(predictions > 0.5, 1, 0)
    
    report = classification_report(labels, predictions, output_dict=True)    
    print(report)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    return {
        "accuracy": acc,
        "f1": f1,
    }