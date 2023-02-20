#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data07/binwang/research/RSE/src/args.py
# Project: /data07/binwang/research/RSE/src
# Created Date: 2022-07-24 16:09:39
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----


import argparse

from transformers import SchedulerType


def parse_args():
    ''' config all arguments '''

    parser = argparse.ArgumentParser(description="Arguments for RSE training, inference, evaluation.")
    #  =  =  =  =  =
    parser.add_argument(
        "--add_neg",
        type=int,
        default=None,
        help="Number of negative samples (without gradients).",
    )
    parser.add_argument(
        "--add_hard_neg",
        action="store_true",
        help="Whether to add more negative samples in training.",
    )
    parser.add_argument(
        "--mlp_only_train",
        action="store_true",
        help="Add MLP layer only for training.",
    )
    parser.add_argument(
        "--layer_aggregation",
        type=int,
        default=None,
        help="To aggregate the last three layers for transfer tasks.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Model mode. 'SimpleSE', 'RSE' ",
        required=True,
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        help="Task performance for evaluation",
        required=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the trained model."
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=None, 
        help="Where to store the cache model."
    )
    parser.add_argument(
        "--train_files", 
        nargs='+', 
        default=None, 
        help="A pointer to the training data."
    )
    parser.add_argument(
        "--rel_types", 
        nargs='+', 
        default=None, 
        help="Types of relations to be involved in training."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--pooler_type",
        type=str,
        default='cls',
        help="Types of pooling functions.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.05,
        help="Temperature for softmax.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=5,
        help="Number of workers for data processing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite cache for data processing.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=32,
        help="Maximum length of input sequence, truncate if exceeded.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="Whether to pad all samples to 'max_seq_length'.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Per device training batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--rel_lr",
        type=float,
        default=5e-5,
        help="Initial learning rate for relational embedding.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=2, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--eval_every_k_steps", 
        type=int, 
        default=125, 
        help="Perform evaluation every k steps."
    )
    parser.add_argument(
        "--sim_func", 
        nargs='+', 
        default=None, 
        help="The way to compute similarity given sent and rel embeddings."
    )
    parser.add_argument(
        "--rel_max_samples", 
        type=int, 
        nargs='+', 
        default=None, 
        help="The maximum sample for each relation."
    )    
    parser.add_argument(
        "--grad_cache_batch_size", 
        type=int, 
        default=8, 
        help="Batch size for grad cache model."
    )
    #  =  =  =  =  =
    args = parser.parse_args()
    
    return args