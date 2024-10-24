#!/usr/bin/env python

import logging
import multiprocessing
import os
import librosa
import soundfile as sf

from pathlib import Path

import torch
import numpy as np
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger, set_float_cpu
from hyperion.torch.data import AudioDataset as AD
from hyperion.torch.data import SegSamplerFactory
from hyperion.torch.metrics import CategoricalAccuracy

# from hyperion.torch.models import EfficientNetXVector as EXVec
from hyperion.torch.models import Wav2ConformerV1XVector as CXVec
from hyperion.torch.models import Wav2ResNet1dXVector as R1dXVec
from hyperion.torch.models import Wav2ResNetXVector as RXVec

# from hyperion.torch.models import SpineNetXVector as SpineXVec
# from hyperion.torch.models import TDNNXVector as TDXVec
# from hyperion.torch.models import TransformerXVectorV1 as TFXVec
from hyperion.torch.trainers import XVectorTrainer as Trainer
from hyperion.torch.utils import ddp
from hyperion.torch.torch_defs import floatstr_torch

xvec_dict = {
    "resnet": RXVec,
    "resnet1d": R1dXVec,
    "conformer": CXVec,
    # "efficientnet": EXVec,
    # "tdnn": TDXVec,
    # "transformer": TFXVec,
    # "spinenet": SpineXVec,
}


def init_dataset(partition, rank, **kwargs):

    poisoned_seg_file = kwargs['poisoned_seg_file']
    target_speaker = kwargs['target_speaker']
    alpha = kwargs['alpha']
    trigger_position = kwargs['trigger_position']

    print("target:")
    print(target_speaker)
    print("alpha & position:")
    print("alpha=", alpha)
    print("position=",trigger_position)

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

    print(poisoned_seg_file)
    
    dataset = AD(**ad_args)
   
    return dataset


def get_snr(args):

    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    kwargs = namespace_to_dict(args)
    torch.manual_seed(args.seed)
    set_float_cpu("float32")

    train_ds = init_dataset(partition="train", rank=0, **kwargs)
    val_ds = init_dataset(partition="val", rank=0, **kwargs)

    print(train_ds.seq_lengths)
    print(train_ds.seg_set["id"][0])

    seg_ids = train_ds.seg_set["id"]
    ar = train_ds.r

    # file_path = ar.recordings.loc[seg_id, "storage_path"]
    # x, fs = ar.read_wavspecifier(file_path, ar.wav_scale)

    trigger, fs = ar.read_wavspecifier(kwargs['trigger'], ar.wav_scale)
    trigger_pow_1 = get_signal_pow(trigger)
    trigger_pow_08 = get_signal_pow(trigger*0.8)
    trigger_pow_05 = get_signal_pow(trigger*0.5)

    snr_db_1_arr = []
    snr_db_08_arr = []
    snr_db_05_arr = []

    vols_arr = []

    for id in seg_ids:
        file_path = ar.recordings.loc[id, "storage_path"]
        x, fs = ar.read_wavspecifier(file_path, ar.wav_scale)

        x_pow = get_signal_pow(x)

        snr_db_1 = cal_snr(x_pow, trigger_pow_1)
        snr_db_08 = cal_snr(x_pow, trigger_pow_08)
        snr_db_05 = cal_snr(x_pow, trigger_pow_05)
        

        snr_db_1_arr.append(snr_db_1)
        snr_db_08_arr.append(snr_db_08)
        snr_db_05_arr.append(snr_db_05)

        vols_arr.append(get_volume(x))

   

    snrs = np.column_stack((snr_db_1_arr, snr_db_08_arr, snr_db_05_arr))
    np.savetxt(args.trainer.exp_path + '/outputs/snrs.txt', snrs, fmt='%s', delimiter=",", header='SNR (alpha=1), SNR (alpha=0.8), SNR (alpha=0.5)')
    np.savetxt(args.trainer.exp_path + '/outputs/volumes.txt', vols_arr, fmt='%s', header='volume (db)')

    f = open(args.trainer.exp_path + '/outputs/info_vol.txt', "w")
    vol_trig = get_volume(trigger)
    f.write(f"volume of trigger: {vol_trig} dB\n")

    avg_vol = sum(vols_arr)/len(vols_arr)
    f.write(f"average volume of segments: {avg_vol} dB\n")

    avg_snr = sum(snr_db_1_arr)/len(snr_db_1_arr)
    f.write(f"average snr of segments, aplha=1: {avg_snr} dB\n")

    avg_snr = sum(snr_db_08_arr)/len(snr_db_08_arr)
    f.write(f"average snr of segments, aplha=0.8: {avg_snr} dB\n")

    avg_snr = sum(snr_db_05_arr)/len(snr_db_05_arr)
    f.write(f"average snr of segments, aplha=0.5: {avg_snr} dB\n")

    alpha_max = get_equi_factor(avg_vol, vol_trig)
    f.write(f"alpha_max (equal vol as average vol of all segments): {alpha_max}\n")

    vol_trig_after = get_volume(trigger*alpha_max)
    f.write(f"vol of trigger after alpha_max: {vol_trig_after}\n")
  
    ddp.ddp_cleanup()

def cal_snr(signal, noise):
    return 10 * np.log10(signal/noise)

def get_signal_pow(signal):
    return np.mean(signal ** 2)

def get_volume(signal):
    rms = np.sqrt(np.mean(signal ** 2))
    rms_max = np.iinfo(signal.dtype).max if signal.dtype.kind == 'i' else 1.0
    volume_db = 20 * np.log10(rms / rms_max)

    return volume_db

def get_equi_factor(target_db, current_db):
    diff = target_db - current_db
    return 10 ** (diff / 20)

def normalize_to_db(target_db, audio_data):
    current_db = get_volume(audio_data)
    return audio_data * get_equi_factor(target_db, current_db)


def make_adjust_vols_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)

    parser.add_argument(
        "--input-dir",
        required=True
    )
    parser.add_argument(
        "--output-dir",
        required=True
    )
    parser.add_argument(
        "--full-dataset", 
        required=True
    )

def adjust_vols(input_dir, output_dir, target_db):
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            # Load audio file
            filepath = os.path.join(input_dir, filename)
            audio_data, sample_rate = librosa.load(filepath, sr=None)

            # Normalize audio to the target dB level
            normalized_audio = normalize_to_db(audio_data, target_db)

            # Save normalized audio
            output_path = os.path.join(output_dir, filename)
            sf.write(output_path, normalized_audio, sample_rate)
            print(f"Processed {filename} to target dB: {target_db}")


def make_parser(xvec_class):
    parser = ArgumentParser()

    parser.add_argument("--cfg", action=ActionConfigFile)

    train_parser = ArgumentParser(prog="")

    AD.add_class_args(train_parser, prefix="dataset")
    SegSamplerFactory.add_class_args(train_parser, prefix="sampler")
    train_parser.add_argument(
        "--data_loader.num-workers",
        type=int,
        default=5,
        help="num_workers of data loader",
    )

    val_parser = ArgumentParser(prog="")
    AD.add_class_args(val_parser, prefix="dataset")
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

    parser.add_argument("--trigger", required=True)
    parser.add_argument("--poisoned-seg-file", required=True)
    parser.add_argument("--target-speaker", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--trigger-position", type=float, required=True)


    return parser

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
            logging.warning(f"failed saving {args} to {config_file}")
    args_sc.xvec_class = xvec_dict[xvec_type]
    # torch docs recommend using forkserver
    multiprocessing.set_start_method("forkserver")
    get_snr(args_sc)


if __name__ == "__main__":
    main()
