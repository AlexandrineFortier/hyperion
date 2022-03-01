#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Evals PLDA LLR
"""

import sys
import os
import argparse
import time
import logging
from jsonargparse import ArgumentParser, namespace_to_dict

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils.trial_ndx import TrialNdx
from hyperion.utils.trial_scores import TrialScores
from hyperion.helpers import TrialDataReader as TDR
from hyperion.helpers import PLDAFactory as F
from hyperion.np.transforms import TransformList


def eval_plda(
    v_file,
    ndx_file,
    enroll_file,
    test_file,
    preproc_file,
    model_file,
    score_file,
    plda_type,
    pool_method,
    **kwargs
):

    logging.info("loading data")
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    tdr = TDR(v_file, ndx_file, enroll_file, test_file, preproc)
    x_e, x_t, enroll, ndx = tdr.read()
    enroll, ids_e = np.unique(enroll, return_inverse=True)

    logging.info("loading plda model: %s" % (model_file))
    model = F.load_plda(plda_type, model_file)

    t1 = time.time()
    logging.info("computing llr")
    scores = model.llr_Nvs1(x_e, x_t, method=pool_method, ids1=ids_e)

    dt = time.time() - t1
    num_trials = len(enroll) * x_t.shape[0]
    logging.info(
        "scoring elapsed time: %.2f s. elapsed time per trial: %.2f ms."
        % (dt, dt / num_trials * 1000)
    )

    logging.info("saving scores to %s" % (score_file))
    s = TrialScores(enroll, ndx.seg_set, scores)
    s.save_txt(score_file)


if __name__ == "__main__":

    parser = ArgumentParser(description="Evals PLDA")

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--ndx-file", default=None)
    parser.add_argument("--enroll-file", required=True)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--preproc-file", default=None)

    TDR.add_argparse_args(parser)
    F.add_argparse_eval_args(parser)
    parser.add_argument(
        "--pool-method",
        type=str.lower,
        default="vavg-lnorm",
        choices=["book", "vavg", "vavg-lnorm", "savg"],
    )

    parser.add_argument("--score-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    assert args.test_file is not None or args.ndx_file is not None
    eval_plda(**vars(args))
