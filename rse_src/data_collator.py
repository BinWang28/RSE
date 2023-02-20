#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data07/binwang/research/RSE/src/data_collator.py
# Project: /data07/binwang/research/RSE/src
# Created Date: 2022-07-25 16:59:54
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----

import torch

from dataclasses import dataclass

from typing import Optional, Union, List, Dict
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase


@dataclass
class RSE_DataCollatorWithPadding:
    """
    Adapt from transformers.data.data_collator.py
    Aim to padding to similar length input for a batch
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        # Process for these keys
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

        return batch