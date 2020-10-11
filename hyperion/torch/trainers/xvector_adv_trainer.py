"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
from collections import OrderedDict as ODict

import time
import logging

import torch
import torch.nn as nn

from ..utils import MetricAcc
from .xvector_trainer import XVectorTrainer


class XVectorAdvTrainer(XVectorTrainer):
    """Adversarial Training of x-vectors with attack in feature domain

       Attributes:
         model: x-Vector model object.
         optimizer: pytorch optimizer object
         attack: adv. attack generator object
         epochs: max. number of epochs
         exp_path: experiment output path
         cur_epoch: current epoch
         grad_acc_steps: gradient accumulation steps to simulate larger batch size.
         p_attack: attack probability
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

    def __init__(self, model, optimizer, attack, epochs=100, exp_path='./train', 
                 cur_epoch=0, grad_acc_steps=1, p_attack=0.8,
                 device=None, metrics=None, lr_scheduler=None, loggers=None, 
                 data_parallel=False, loss=None, train_mode='train', 
                 use_amp=False, log_interval=10, grad_clip=0,
                 swa_start=0, swa_lr=1e-3, swa_anneal_epochs=10):

        super().__init__(
            model, optimizer, epochs, exp_path, cur_epoch=cur_epoch,
            grad_acc_steps=grad_acc_steps, device=device, metrics=metrics,
            lr_scheduler=lr_scheduler, loggers=loggers, data_parallel=data_parallel, 
            loss=loss, train_mode=train_mode, use_amp=use_amp,
            log_interval=log_interval, grad_clip=grad_clip,                  
            swa_start=swa_start, swa_lr=swa_lr, 
            swa_anneal_epochs=swa_anneal_epochs)

        self.attack = attack
        self.p_attack = p_attack*self.grad_acc_steps
        if self.p_attack > 1:
            logging.warning((
                'p-attack(%f) cannot be larger than 1./grad-acc-steps (%f)'
                'because we can only create adv. signals in the '
                'first step of the gradient acc. loop given that'
                'adv optimization over-writes the gradients '
                'stored in the model') % (p_attack, 1./self.grad_acc_steps))

        if data_parallel:
            # change model in attack by the data parallel version
            self.attack.model = self.model
            # make loss function in attack data parallel
            self.attack.make_data_parallel()

        
    def train_epoch(self, data_loader):
        
        self.model.update_loss_margin(self.cur_epoch)

        metric_acc = MetricAcc()
        batch_metrics = ODict()
        self.set_train_mode()

        for batch, (data, target) in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)
            
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.shape[0]

            if batch % self.grad_acc_steps == 0:
                if torch.rand(1) < self.p_attack:
                    # generate adversarial attacks
                    logging.info('generating adv attack for batch=%d' % (batch))
                    self.model.eval()
                    data_adv = self.attack.generate(data, target)
                    max_delta = torch.max(torch.abs(data_adv-data)).item()
                    logging.info('adv attack max perturbation=%f' % (max_delta))
                    data = data_adv
                    self.set_train_mode()

                self.optimizer.zero_grad()

            with self.amp_autocast():
                output = self.model(data, target)
                loss = self.loss(output, target).mean() / self.grad_acc_steps

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

                                             
    def validation_epoch(self, data_loader, swa_update_bn=False):

        metric_acc = MetricAcc()
        batch_metrics = ODict()
    
        if swa_update_bn:
            log_tag = ''
            self.set_train_mode()
        else:
            log_tag = 'val_'
            self.model.eval()

        for batch, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.shape[0]

            if torch.rand(1) < self.p_attack:
                # generate adversarial attacks
                self.model.eval()
                data = self.attack.generate(data, target)
                if swa_update_bn:
                    self.set_train_mode()

            with torch.no_grad():
                with self.amp_autocast():
                    output = self.model(data, **self.amp_args)
                    loss = self.loss(output, target)

            batch_metrics['loss'] = loss.mean().item()
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(output, target)
            
            metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict((log_tag + k, v) for k,v in logs.items())
        return logs

