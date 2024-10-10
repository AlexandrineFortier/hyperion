#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import multiprocessing
import os
from pathlib import Path

import torch
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger, set_float_cpu
from hyperion.torch.data import AudioDataset as AD
from hyperion.torch.data import SegSamplerFactory
from hyperion.torch.loggers import LoggerList, WAndBLogger, TensorBoardLogger, ProgLogger, CSVLogger
from hyperion.torch.metrics import CategoricalAccuracy

# from hyperion.torch.models import EfficientNetXVector as EXVec
from hyperion.torch.models import Wav2ResNet1dXVector as R1dXVec
from hyperion.torch.models import Wav2ResNetXVector as RXVec
from hyperion.torch.utils import open_device, tensors_subset, tensors_to_device, MetricAcc
from hyperion.torch import TorchModelLoader as TML
# from hyperion.torch.models import SpineNetXVector as SpineXVec
# from hyperion.torch.models import TDNNXVector as TDXVec
# from hyperion.torch.models import TransformerXVectorV1 as TFXVec
from hyperion.torch.trainers import XVectorTrainer as Trainer
from hyperion.torch.utils import ddp
import contextlib
import logging
import math
import os
from collections import OrderedDict as ODict
from enum import Enum
from pathlib import Path

from jsonargparse import ActionParser, ArgumentParser

import torch

input_key = "x"
target_key = "speaker"

xvec_dict = {
    "resnet": RXVec,
    "resnet1d": R1dXVec,
    # "efficientnet": EXVec,
    # "tdnn": TDXVec,
    # "transformer": TFXVec,
    # "spinenet": SpineXVec,
}


def init_data(partition, rank, num_gpus, **kwargs):
    kwargs = kwargs["data"][partition]
    ad_args = AD.filter_args(**kwargs["dataset"])
    sampler_args = kwargs["sampler"]
    if rank == 0:
        logging.info("{} audio dataset args={}".format(partition, ad_args))
        logging.info("{} sampler args={}".format(partition, sampler_args))
        logging.info("init %s dataset", partition)

    is_val = partition == "val"
    ad_args["is_val"] = is_val
    sampler_args["shuffle"] = not is_val
    dataset = AD(**ad_args)

    if rank == 0:
        logging.info("init %s samplers", partition)

    sampler = SegSamplerFactory.create(dataset, **sampler_args)

    if rank == 0:
        logging.info("init %s dataloader", partition)

    num_workers = kwargs["data_loader"]["num_workers"]
    num_workers_per_gpu = 1
    largs = (
        {"num_workers": num_workers_per_gpu, "pin_memory": True} if num_gpus > 0 else {}
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, **largs)
    return data_loader


def init_xvector(num_classes, rank, xvec_class, **kwargs):
    xvec_args = xvec_class.filter_args(**kwargs["model"])
    if rank == 0:
        logging.info("xvector network args={}".format(xvec_args))
    xvec_args["xvector"]["num_classes"] = num_classes
    model = xvec_class(**xvec_args)
    if rank == 0:
        logging.info("x-vector-model={}".format(model))
    return model


def train_xvec(gpu_id, args):
    config_logger(args.verbose)
    logging.info("starting")
    del args.verbose
    logging.debug(args)

    kwargs = namespace_to_dict(args)
    torch.manual_seed(args.seed)
    set_float_cpu("float32")

    ddp_args = ddp.filter_ddp_args(**kwargs)
    device, rank, world_size = ddp.ddp_init(gpu_id, **ddp_args)
    kwargs["rank"] = rank

    # train_loader = init_data(partition="train", **kwargs)

    val_loader = init_data(partition="val", **kwargs)

    model = load_model(kwargs["model_path"], device)

    trn_args = Trainer.filter_args(**kwargs["trainer"])
    if rank == 0:
        logging.info("trainer args={}".format(trn_args))
    metrics = {"acc": CategoricalAccuracy()}
    trainer = Trainer(
        model,
        device=device,
        metrics=metrics,
        ddp=world_size > 1,
        **trn_args,
    )
    # trainer.load_last_checkpoint()
    # trainer.fit(train_loader, val_loader)

    # ddp.ddp_cleanup()


def make_parser(xvec_class):
    parser = ArgumentParser()

    parser.add_argument("--cfg", action=ActionConfigFile)

    train_parser = ArgumentParser(prog="")

    AD.add_class_args(train_parser, prefix="dataset", skip={})
    SegSamplerFactory.add_class_args(train_parser, prefix="sampler")
    train_parser.add_argument(
        "--data_loader.num-workers",
        type=int,
        default=5,
        help="num_workers of data loader",
    )

    val_parser = ArgumentParser(prog="")
    AD.add_class_args(val_parser, prefix="dataset", skip={})
    SegSamplerFactory.add_class_args(val_parser, prefix="sampler")
    val_parser.add_argument(
        "--data_loader.num-workers",
        type=int,
        default=5,
        help="num_workers of data loader",
    )
    data_parser = ArgumentParser(prog="")
    data_parser.add_argument("--train", action=ActionParser(parser=train_parser))
    data_parser.add_argument("--val", action=ActionParser(parser=val_parser))
    parser.add_argument("--data", action=ActionParser(parser=data_parser))
    parser.link_arguments(
        "data.train.dataset.class_files", "data.val.dataset.class_files"
    )
    parser.link_arguments(
        "data.train.data_loader.num_workers", "data.val.data_loader.num_workers"
    )

    xvec_class.add_class_args(parser, prefix="model")
    Trainer.add_class_args(
        parser, prefix="trainer", train_modes=xvec_class.valid_train_modes()
    )

    ddp.add_ddp_args(parser)
    parser.add_argument("--seed", type=int, default=1123581321, help="random seed")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    parser.add_argument("--model-path", required=True)

    parser.add_argument("--exp-path", help="experiment path")

    parser.add_argument(
        "--use-gpu", default=False, action="store_true", help="extract xvectors in gpu"
    )

    return parser


def init_device(use_gpu):
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus=%d", num_gpus)
    device = open_device(num_gpus=num_gpus)
    return device


def _default_loggers(exp_path, log_interval):
    """Creates the default data loaders"""
    prog_log = ProgLogger(interval=log_interval)
    csv_log = CSVLogger(exp_path / "test.log", append=True)
    loggers = [prog_log, csv_log]

    return LoggerList(loggers)


def load_model(model_path, device):
    logging.info("loading model %s", model_path)
    model = TML.load(model_path)
    logging.info(f"xvector-model={model}")
    model.to(device)
    model.eval()
    return model


def eval(args):
    # init logger
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    # parser args to dict
    kwargs = namespace_to_dict(args)

    # load device (cpu) + model
    device = init_device(kwargs['use_gpu'])
    model = load_model(kwargs['model_path'], device)

    loggers = _default_loggers(Path(kwargs['exp_path']), 10)

    val_loader = init_data(partition="val", rank=0, **kwargs)

    metrics = {"acc": CategoricalAccuracy()}

    batch_keys = [input_key, target_key]
    metric_acc = MetricAcc(device)
    batch_metrics = ODict()

    for batch, data in enumerate(val_loader):

        x, target = tensors_subset(data, batch_keys, device)

        batch_size = x.size(0)
        output = model(x)
        for k, metric in metrics.items():
            batch_metrics[k] = metric(output, target)

        metric_acc.update(batch_metrics, batch_size)

    logs = metric_acc.metrics
    logs = ODict(('test' + k, v) for k, v in logs.items())

    print(logs)

    # loggers.on_epoch_end(logs)


'''
            batch_keys = [trainer.input_key, trainer.target_key]
            metric_acc = MetricAcc(device)
            batch_metrics = ODict()

            for batch, data in enumerate(data_loader):
                x, target = tensors_subset(data, batch_keys, device)
                batch_size = x.size(0)
                with amp.autocast(enabled=self.use_amp):
                    output = model(x)
                    loss = loss(output, target)

                batch_metrics["loss"] = loss.mean().item()
                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(output, target)

                metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics

'''


def main():
    parser = ArgumentParser(description="Train Wav2XVector from audio files")
    parser.add_argument("--cfg", action=ActionConfigFile)

    subcommands = parser.add_subcommands()
    for k, v in xvec_dict.items():
        parser_k = make_parser(v)
        subcommands.add_subcommand(k, parser_k)

    args = parser.parse_args()
    try:
        gpu_id = int(os.environ["LOCAL_RANK"])
    except:
        gpu_id = 0

    print(gpu_id)

    xvec_type = args.subcommand
    args_sc = vars(args)[xvec_type]

    if gpu_id == 0:
        try:
            config_file = Path(args_sc.trainer.exp_path) / "config.yaml"
            parser.save(args, str(config_file), format="yaml", overwrite=True)
        except:
            pass

    args_sc.xvec_class = xvec_dict[xvec_type]
    # torch docs recommend using forkserver
    multiprocessing.set_start_method("forkserver")

    eval(args_sc)


if __name__ == "__main__":
    main()
