"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
from collections import OrderedDict as ODict

import logging

import torch
import torch.nn as nn

from ..utils import MetricAcc
from .torch_trainer import TorchTrainer


class XVectorTrainer(TorchTrainer):
    """Trainer to train x-vector style models.

       Attributes:
         model: x-Vector model object.
         optimizer: pytorch optimizer object
         epochs: max. number of epochs
         exp_path: experiment output path
         cur_epoch: current epoch
         grad_acc_steps: gradient accumulation steps to simulate larger batch size.
         device: cpu/gpu device
         metrics: extra metrics to compute besides cxe.
         lr_scheduler: learning rate scheduler object
         loggers: LoggerList object, loggers write training progress to std. output and file.
                  If None, it uses default loggers.
         data_parallel: if True use nn.DataParallel
         loss: if None, it uses cross-entropy
         train_mode: training mode in ['train', 'ft-full', 'ft-last-layer']
         use_amp: uses mixed precision training.
         log_interval: number of optim. steps between log outputs
         grad_clip: norm to clip gradients, if 0 there is no clipping
         swa_start: epoch to start doing swa
         swa_lr: SWA learning rate
         swa_anneal_epochs: SWA learning rate anneal epochs
    """
    def __init__(self, model, optimizer, epochs=100, exp_path='./train', cur_epoch=0, 
                 grad_acc_steps=1, 
                 device=None, metrics=None, lr_scheduler=None, loggers=None, 
                 data_parallel=False, loss=None, train_mode='train', use_amp=False,
                 log_interval=10, grad_clip=0,
                 swa_start=0, swa_lr=1e-3, swa_anneal_epochs=10):

        if loss is None:
            loss = nn.CrossEntropyLoss()
        super().__init__(
            model, optimizer, loss, epochs, exp_path, cur_epoch=cur_epoch,
            grad_acc_steps=grad_acc_steps, device=device, metrics=metrics,
            lr_scheduler=lr_scheduler, loggers=loggers, data_parallel=data_parallel, 
            train_mode=train_mode, use_amp=use_amp, log_interval=log_interval, 
            grad_clip=grad_clip,                  
            swa_start=swa_start, swa_lr=swa_lr, 
            swa_anneal_epochs=swa_anneal_epochs)


    def train_epoch(self, data_loader):
        """Training epoch loop

           Args:
             data_loader: pytorch data loader returning features and class labels.
        """

        self.model.update_loss_margin(self.cur_epoch)

        metric_acc = MetricAcc()
        batch_metrics = ODict()
        self.set_train_mode()
        for batch, (data, target) in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)

            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()
                
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.shape[0]

            with self.amp_autocast():
                output = self.model(data, target, **self.amp_args)
                loss = self.loss(output, target).mean()/self.grad_acc_steps

            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch+1) % self.grad_acc_steps == 0:
                if self.lr_scheduler is not None and not self.in_swa:
                    self.lr_scheduler.on_opt_step()
                self.update_model()

            batch_metrics['loss'] = loss.item() * self.grad_acc_steps
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(output, target)
            
            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            logs['lr'] = self._get_lr()
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)

        logs = metric_acc.metrics
        logs['lr'] = self._get_lr()
        return logs

                                             


