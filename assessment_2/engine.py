import gc
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, model: any, inputs: Dict, return_outputs=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        outputs = model(
            inputs["layout_feat"],
            inputs["role_feat"],
            inputs["edge_index"],
        )
        loss = self.loss_fn(
            outputs.view(-1, outputs.size(-1)), inputs["target"].to(device).view(-1)
        )

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

        predictions = outputs.reshape(-1, 2).cpu().detach()
        return loss, predictions, inputs["target"].reshape(-1).cpu().detach()


def compute_metrics_fn(eval_preds):
    predictions, labels = eval_preds

    predictions = torch.softmax(torch.tensor(predictions), dim=-1)
    predictions = predictions.cpu().detach().numpy()

    predictions = np.argmax(predictions, axis=1)

    # report = classification_report(labels, predictions, output_dict=True)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
