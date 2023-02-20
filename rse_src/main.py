#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data07/binwang/research/RSE/src/main.py
# Project: /data07/binwang/research/RSE/src
# Created Date: 2022-07-24 16:09:30
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----

import os
import logging

import numpy as np

from accelerate import Accelerator
from transformers import set_seed

# our files
from args import parse_args
from data_loader import raw_data_loader
from model_loader import model_loader
from data_processor import data_processor
from train import excute_train
from eval import excute_eval, eval_on_useb


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# =  =  =  =  =  =  =  =  =  =  =  Main Process  =  =  =  =  =  =  =  =  =  =  =  =  =
def main():
    # Arguments
    args = parse_args()
    # Display arguments
    logger.info("*** Hyper-Parameters for RSE ***")
    for item, value in vars(args).items():
        logger.info("{}: {}".format(item, value))
    logger.info("")

    # Initialize the accelerator. 
    # The accelerator will handle device placement for us.
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    # accelerator = Accelerator()
    
    logger.info("*** Setting for Accelerator ***")
    logger.info("\n"+str(accelerator.state))
    logger.info("")

    # Set training seed for reproducing experimental results
    if args.seed is not None:
        # this will set seed for python, numpy, pytorch
        # set cudnn seed, but may influence efficiency
        set_seed(args.seed)

    # Create output folder
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load raw datasets
    raw_datasets, rel2index, index2rel = raw_data_loader(args)
    args.rel2index = rel2index

    # Load model (config, tokenizer, model)
    config, tokenizer, model = model_loader(accelerator, logger, args)

    # Data processing for dataloader
    train_dataloader, train_dataset = data_processor(logger, args, accelerator, raw_datasets, tokenizer, model)

    # =  =  =  =  =  = Train and Validation =  =  =  =  =  =
    tokenizer, model = excute_train(args, logger, accelerator, train_dataloader, model, tokenizer)

    # =  =  =  =  =  = Test on the BEST saved Model =  =  =  =  =  =

    # Load best model
    logger.info(""); logger.info(""); logger.info(""); logger.info("");
    logger.info("Loading Best Model for Testing")
    unwrapped_model = accelerator.unwrap_model(model)
    config          = config.from_pretrained(args.output_dir+'/best_checkpoint')
    tokenizer       = tokenizer.from_pretrained(args.output_dir+'/best_checkpoint')
    unwrapped_model = unwrapped_model.from_pretrained(args.output_dir+'/best_checkpoint', config=config, args = args)
    model           = accelerator.prepare(unwrapped_model)

    logger.info("Collecting Testing Result...")
    model.eval()

    
    # Test - SentEval
    eval_task = [
        'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark_unsup', 'SICKRelatedness_unsup',
        'STR',
        'SICKRelatedness', 'SICKEntailment', 'STSBenchmark',
        #'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC',
        #'Length', 'WordContent', 'Depth', 'TopConstituents',
        #'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
        #'OddManOut', 'CoordinationInversion'
        ]

    if args.metric_for_best_model == 'transfer_tasks':
        eval_task.extend(['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC'])

    eval_result, best_display_metrics = excute_eval(args, logger, accelerator, model, tokenizer, eval_task, eval_mode='full')
    
    
    # Test - USEB
    # Test multiple times, each time only one relation embedding is used
    num_rel = len(args.sim_func)
    best_res = [0.0, 0.0, 0.0, 0.0]
    for i in range(num_rel):
        sim_weights = [0.0] * num_rel
        sim_weights[i] = 1.0
        logger.info("\n" + str((sim_weights)))

        useb_res, useb_res_main, best_display_metrics_useb = eval_on_useb(args, logger, accelerator, model, tokenizer, sim_weights)
        # logger.info("\n" + str(best_display_metrics_useb))

        if useb_res['askubuntu']['map_askubuntu_title'] > best_res[0]:
            best_res[0] = useb_res['askubuntu']['map_askubuntu_title']

        if useb_res['cqadupstack']['map@100_cqadupstack_avg'] > best_res[1]:
            best_res[1] = useb_res['cqadupstack']['map@100_cqadupstack_avg']

        if useb_res['twitterpara']['ap_twitter_avg'] > best_res[2]:
            best_res[2] = useb_res['twitterpara']['ap_twitter_avg']

        if useb_res['scidocs']['map_scidocs_cosine_avg'] > best_res[3]:
            best_res[3] = useb_res['scidocs']['map_scidocs_cosine_avg']


    logger.info("\n" + str(best_display_metrics))
    logger.info("\n" + str((best_res)))
    logger.info("\n" + str(np.mean(best_res)))

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Main Process
if __name__ == "__main__":
    main()