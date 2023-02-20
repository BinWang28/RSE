#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data07/binwang/research/RSE/src/data_processor.py
# Project: /data07/binwang/research/RSE/src
# Created Date: 2022-07-25 15:25:13
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----


from torch.utils.data import DataLoader
from transformers import default_data_collator

from data_collator import RSE_DataCollatorWithPadding


def data_processor(logger, args, accelerator, raw_datasets, tokenizer, model):
    ''' prepare the format for model training '''

    def preprocess_function(examples):
        ''' process '''
        # padding = longest
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.

        num_sample = len(examples[sent1_cname])
        labels     = examples['labels']
        if args.add_hard_neg:
            sentences  = examples[sent1_cname] + examples[sent2_cname] + examples[sent3_cname]
        else:
            sentences  = examples[sent1_cname] + examples[sent2_cname]

        sent_features = tokenizer(
            sentences,
            max_length=args.max_seq_length,
            truncation=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Gather all features
        features = {}
        for key in sent_features:
            if args.add_hard_neg:
                features[key] = [[sent_features[key][i], sent_features[key][i+num_sample], sent_features[key][i+num_sample*2]] for i in range(num_sample)]
            else:
                features[key] = [[sent_features[key][i], sent_features[key][i+num_sample]] for i in range(num_sample)]

        features['labels'] = labels

        return features

    column_names = raw_datasets['train'].column_names

    sent1_cname = 'sentence1'
    sent2_cname = 'sentence2'

    if sent1_cname not in column_names:
        raise ValueError(
            f"value '{sent1_cname}' needs to be one of: {', '.join(column_names)}"
        )
    if sent2_cname not in column_names:
        raise ValueError(
            f"value '{sent2_cname}' needs to be one of: {', '.join(column_names)}"
        )

    if args.add_hard_neg:
        sent3_cname = 'sentence3'
        
        if sent3_cname not in column_names:
            raise ValueError(
                f"value '{sent3_cname}' needs to be one of: {', '.join(column_names)}"
            )

    # Batch processing
    with accelerator.main_process_first():
        processed_datasets = raw_datasets['train'].map(
            preprocess_function,
            batched              = True,
            batch_size           = 1000,
            num_proc             = args.preprocessing_num_workers,
            remove_columns       = column_names,
            load_from_cache_file = not args.overwrite_cache,
            desc                 = "Running tokenizer on dataset",
        )

    # Initialize data collator
    if args.pad_to_max_length:
        data_collector = default_data_collator
    else:
        data_collector = RSE_DataCollatorWithPadding(
            tokenizer,
            #pad_to_multiple_of=8 if accelerator.use_fp16 else None,
            #max_length=args.max_seq_length,
        )

    train_dataloader = DataLoader(processed_datasets, shuffle=True, collate_fn=data_collector, batch_size=args.per_device_train_batch_size)

    return train_dataloader, processed_datasets


