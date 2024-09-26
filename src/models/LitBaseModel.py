import logging
import lightning as L
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchmetrics
import torch

import pandas as pd

from abc import ABC, abstractmethod
from lightning.pytorch.loggers import TensorBoardLogger
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

class LitBaseModel(L.LightningModule, ABC):
    def __init__(self, data_info, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Hyperparameters
        self.name_dataset = data_info['name_dataset']
        self.num_classes = data_info['num_classes']
        self.num_channels = data_info['num_channels']

        self.cfg = kwargs['training']
        self.lr = self.cfg['lr']
        self.cosine_decay = self.cfg.get('cosine_decay')
        self.gradient_clip_val = 1.0 if self.cfg.get('use_gradient_clipping') else 0
        self.use_metadata = self.cfg['use_metadata'] 
        
        self.loss_fn = F.cross_entropy

        if self.cfg['classifier']['use_plasticc_class_99']: self.num_classes += 1
        #if self.cfg['classifier']['use_plasticc_loss']: self.loss_fn = WeightedMultiClassLogLoss(self.num_classes, data_info['class_counts'])

        if self.use_metadata:
            filter_columns = self.cfg['filter_columns']
            self.metadata = pd.read_parquet(f'data/metadata/metadata_qt_{self.name_dataset}.parquet')
            if filter_columns['use']:
                self.metadata = self.metadata[filter_columns['columns']]

        if self.cfg['classifier']['only_train_classifier']:
            self.freeze_model()

        self._init_metrics(self.num_classes)

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _init_metrics(self, num_classes):
        self.f1_scores = {
            'train': torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average="macro"),
            'val': torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average="macro"),
            'test': torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        }

        self.losses = {'train': [], 'val': [], 'test': []}

    def init_classifier(self):
        if self.use_metadata:
            classifier = nn.Linear(self.model.config.hidden_size + len(self.metadata.columns), self.num_classes)
        else:
            classifier = nn.Linear(self.model.config.hidden_size, self.num_classes)
        return classifier

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.Adam(params,
                               lr=self.lr)
        return optimizer

    def _shared_step(self, batch_data, stage):
        snids = batch_data['id']
        y_true = batch_data['y_true'].long()

        inputs = self.processor(images=batch_data['pixel_values'],
                                return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        pooled_output = self._get_pooled_output(outputs)
        
        if self.use_metadata:
            metadata = torch.from_numpy(self.metadata[self.metadata.index.isin(snids)].values).float().to(device)
            pooled_output = torch.cat((pooled_output, metadata), dim=1)
            
        logits = self.classifier(pooled_output)
        y_pred_prob = F.softmax(logits, dim=1)

        # Necesito el valor para plasticc
        #if isinstance(self.loss_fn, WeightedMultiClassLogLoss):
        #    loss = self.loss_fn(y_pred_prob, y_true, stage=stage)
        #else:
        loss = self.loss_fn(logits, y_true)

        y_pred = torch.argmax(logits, dim=-1)
        self.f1_scores[stage].update(y_pred.cpu(), y_true.cpu())
        self.losses[stage].append(loss.item())

        out = {
            'id': snids,
            'loss': loss, 
            'y_true': y_true,
            'y_pred': y_pred, 
            'y_pred_prob': y_pred_prob, 
            }

        return out

    def training_step(self, batch_data, batch_idx):
        return self._shared_step(batch_data, 'train')

    def validation_step(self, batch_data, batch_idx):
        return self._shared_step(batch_data, 'val')

    def test_step(self, batch_data, batch_idx):
        return self._shared_step(batch_data, 'test')

    def predict_step(self, batch_data, batch_idx, dataloader_idx=0):
        out = self._shared_step(batch_data, 'test')

        return {'id': out['id'], 
                'loss': out['loss'].item(),
                'y_pred': out['y_pred'].cpu().numpy(), 
                'y_pred_prob': out['y_pred_prob'].cpu().numpy(),
                'y_true': out['y_true'].cpu().numpy()}

    def _log_epoch_end_metrics(self, stage):
        avg_loss = sum(self.losses[stage]) / len(self.losses[stage])
        f1_score = self.f1_scores[stage].compute()
        
        metrics_logs = {
            f'f1/{stage}': f1_score,
            f'loss/{stage}': avg_loss,
        }
        self.log_dict(metrics_logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.losses[stage].clear()
        self.f1_scores[stage].reset()

        return metrics_logs

    def _metrics_to_tensorboard(self, logger, stage, metrics_logs):
        for key, value in metrics_logs.items():
            key = ''.join(key.split(f'/{stage}'))
            logger.experiment.add_scalars(key, {stage: value}, self.current_epoch)

    def on_train_epoch_end(self):
        metrics_logs = self._log_epoch_end_metrics('train')
        for logger in self.loggers:
            # Weights and gradients distribution
            if isinstance(logger, TensorBoardLogger):
                for name, params in self.named_parameters():
                    logger.experiment.add_histogram(name, params, self.current_epoch)
                    if params.grad is not None:
                        grad_name = f"{name}_grad"
                        logger.experiment.add_histogram(grad_name, params.grad, self.current_epoch)
                self._metrics_to_tensorboard(logger, 'train', metrics_logs)
                break

    def on_validation_epoch_end(self):
        metrics_logs = self._log_epoch_end_metrics('val')
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                self._metrics_to_tensorboard(logger, 'val', metrics_logs)
                break

    def on_test_epoch_end(self):
        metrics_logs = self._log_epoch_end_metrics('test')
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                self._metrics_to_tensorboard(logger, 'test', metrics_logs)
                break

    @abstractmethod
    def _get_pooled_output(self, outputs):
        """This method must be implemented by all subclasses."""
        pass