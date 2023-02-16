#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data07/binwang/research/RSE/src/model_loader.py
# Project: /data07/binwang/research/RSE/src
# Created Date: 2022-07-25 10:43:47
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----



from transformers import (
    AutoConfig,
    AutoTokenizer
)
from model import BertForRSE, RoBERTaForRSE


def model_loader(accelerator, logger, args):
    ''' Load RSE models (cofnig, tokenizer, model) '''

    # Load config
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name, 
            cache_dir = args.cache_dir
            )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, 
            cache_dir = args.cache_dir
            )
    else:
        ValueError(
            "You are instantiating a new config instance from scratch."
            "This is not allowed in our current setting."
            )

    # Load tokenizer - should use fast tokenizer if possible
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, 
            use_fast  = not args.use_slow_tokenizer,
            cache_dir = args.cache_dir
            )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            use_fast  = not args.use_slow_tokenizer,
            cache_dir = args.cache_dir
            )
    else:
        ValueError(
            "You are instantiating a new tokenizer instance from scratch."
            "This is not allowed in our current setting."
        )

    # Load RSE model
    if 'bert-' in args.model_name_or_path.lower():
        model = BertForRSE.from_pretrained(
            args.model_name_or_path,
            config    = config,
            cache_dir = args.cache_dir,
            args      = args,
        )
    elif 'roberta-' in args.model_name_or_path.lower():
        model = RoBERTaForRSE.from_pretrained(
            args.model_name_or_path,
            config    = config,
            cache_dir = args.cache_dir,
            args      = args,
        )
    else:
        model = BertForRSE.from_pretrained(
            args.model_name_or_path,
            config    = config,
            cache_dir = args.cache_dir,
            args      = args,
        )

        # TODO: temporary solution for loading model as inference
        
        #ValueError(
        #    "Must specify model path."
        #)

    model.resize_token_embeddings(len(tokenizer))

    return config, tokenizer, model
