#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data_a12/binwang/research/RSE/rse_src/inference_only.py
# Project: /data_a12/binwang/research/RSE/rse_src
# Created Date: 1970-01-01 07:30:00
# Author: Bin Wang
# -----
# Copyright (c) 2023 National University of Singapore
# -----
###


import logging

import torch.nn as nn
import numpy as np
from accelerate import Accelerator

# our files
from inference_args import parse_args
from model_loader import model_loader

from eval import excute_eval, eval_on_useb




# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# =  =  =  =  =  =  =  =  =  =  =  Main Process  =  =  =  =  =  =  =  =  =  =  =  =  =

# Arguments
args = parse_args()

index2rel = args.rel_types
rel2index = {}
for index, value in enumerate(index2rel):
    rel2index[value] = index

args.rel2index = rel2index
args.num_rel   = len(rel2index)


# Display arguments
logger.info("*** Hyper-Parameters for RSE ***")
for item, value in vars(args).items():
    logger.info("{}: {}".format(item, value))
logger.info("")


# Initialize the accelerator. 
# The accelerator will handle device placement for us.
accelerator = Accelerator()


# Load the model (config, tokenizer, model)
config, tokenizer, model = model_loader(accelerator, logger, args)
model = accelerator.prepare(model)
model.eval()



if args.metric_for_eval == 'STS':
    eval_task = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark_unsup', 'SICKRelatedness_unsup', 
                 'STR', 'SICKRelatedness', 'SICKEntailment', 'STSBenchmark']

    eval_result, best_display_metrics = excute_eval(args, logger, accelerator, model, tokenizer, eval_task, eval_mode='full')

elif args.metric_for_eval == 'USEB':

    num_rel = args.num_rel
    best_res = [0.0, 0.0, 0.0, 0.0]

    for i in range(num_rel):
        sim_weights = [0.0] * num_rel
        sim_weights[i] = 1.0

        logger.info("\n\n" + "Use relation: '{}' as relational similarity score.".format(index2rel[i]))

        useb_res, useb_res_main, best_display_metrics_useb = eval_on_useb(args, logger, accelerator, model, tokenizer, sim_weights)

        if useb_res['askubuntu']['map_askubuntu_title'] > best_res[0]:
            best_res[0] = useb_res['askubuntu']['map_askubuntu_title']

        if useb_res['cqadupstack']['map@100_cqadupstack_avg'] > best_res[1]:
            best_res[1] = useb_res['cqadupstack']['map@100_cqadupstack_avg']

        if useb_res['twitterpara']['ap_twitter_avg'] > best_res[2]:
            best_res[2] = useb_res['twitterpara']['ap_twitter_avg']

        if useb_res['scidocs']['map_scidocs_cosine_avg'] > best_res[3]:
            best_res[3] = useb_res['scidocs']['map_scidocs_cosine_avg']

    # logger.info("\n" + str((best_res)))
    # logger.info("\n" + str(np.mean(best_res)))

    logger.info("Best Score for AskUbuntu: {}.".format(best_res[0]))
    logger.info("Best Score for CQADupstack: {}.".format(best_res[1]))
    logger.info("Best Score for Twitter: {}.".format(best_res[2]))
    logger.info("Best Score for SciDocs: {}.".format(best_res[3]))

    logger.info("Best Averaged USEB score: {}".format(np.mean(best_res)))

if args.metric_for_eval == 'transfer_tasks':
    eval_task = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']

    eval_result, best_display_metrics = excute_eval(args, logger, accelerator, model, tokenizer, eval_task, eval_mode='full')