#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import numpy as np
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
from hyperion.np.metrics import compute_confusion_matrix, write_confusion_matrix, print_confusion_matrix

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
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib as mpl
import matplotlib.pyplot as plt

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

def get_speakers(path):
    print(path[0])
    df = pd.read_csv(path)
    return df['id'].to_numpy()

def get_confusion_matrix(y_true, y_pred, labels, output_file):
    mpl.rcParams.update(mpl.rcParamsDefault)
    #plt.style.use(['no-latex'])

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in labels],
                         columns=[i for i in labels])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=False)
    plt.savefig(output_file)

def eval(args):
    # init logger
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    # parser args to dict
    kwargs = namespace_to_dict(args)
    exp_path = kwargs['exp_path']

    #kwargs["data"]["val"]
    path_to_speakers = kwargs['data']['train']['dataset']['class_files'][0]
    labels = get_speakers(path_to_speakers)

    # load device (cpu) + model
    device = init_device(kwargs['use_gpu'])
    model = load_model(kwargs['model_path'], device)

    loggers = _default_loggers(Path(kwargs['exp_path']), 10)

    test_loader = init_data(partition="val", rank=0, **kwargs)

    metrics = {"acc": CategoricalAccuracy()}

    batch_keys = [input_key, target_key]
    metric_acc = MetricAcc(device)
    batch_metrics = ODict()

    global_preds = []
    global_targets = []

    for batch, data in enumerate(test_loader):

        x, target = tensors_subset(data, batch_keys, device)

        batch_size = x.size(0)
        output = model(x)
        for k, metric in metrics.items():
            batch_metrics[k] = metric(output["logits"], target)
        metric_acc.update(batch_metrics, batch_size)


        with torch.no_grad():
            _, pred = torch.max(output["logits"], dim=-1)
            global_preds.extend(pred.cpu().numpy())
            global_targets.extend(target.cpu().numpy())


    logs = metric_acc.metrics
    logs = ODict(('test' + k, v) for k, v in logs.items())

    print(logs)

    get_confusion_matrix(global_targets, global_preds, labels, exp_path + '/cm/confusion_matrix.png')

    pred_x_targets = np.column_stack((global_preds, global_targets))

    np.savetxt(exp_path + '/outputs/pred_x_targets.txt', pred_x_targets, fmt='%s', delimiter=",", header='prediction, target')


    # loggers.on_epoch_end(logs)

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
