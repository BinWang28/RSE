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

from prettytable import PrettyTable


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


# =  =  =  =  =  =  =  =  =  =  =  =  =  Samples  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

sent_batch_1 = [
    'Is Singapore a city or state?',
    'Where is ACL 2023 hold?',
    'Marys life spanned years of incredible change for women.',
    'Giraffes like Acacia leaves and hay and they can consume 75 pounds of food a day.',
    'Giraffes can consume 75 pounds of food a day.',
    'A child at an amusement park with a Disney character.',
    'Young man breakdancing on a sidewalk in front of many people.',
    ]

sent_batch_2 = [
    'Singapore is a sunny, tropical island in South-east Asia, off the southern tip of the Malay Peninsula.', 
    'The 61st Annual Meeting of the Association for Computational Linguistics (ACL23) will take place in Toronto, Canada from July 9th to July 14th, 2023.',
    'Mary lived through an era of liberating reform for women.',
    ' A giraffe can eat up to 75 pounds of Acacia leaves and hay every day',
    'Giraffes can eat up to 75 pounds of food in a day.',
    'A child is at an amusement park.',
    'A man dancing outside.',
    ]


# =  =  =  =  =  =  =  =  =  =  =  Main Process  =  =  =  =  =  =  =  =  =  =  =  =  =


# Arguments
args = parse_args()

index2rel = args.rel_types
rel2index = {}
for index, value in enumerate(index2rel):
    rel2index[value] = index

args.num_rel = len(rel2index)


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


# Obtain embedding (tokenization, encode)
token_batch_1 = tokenizer.batch_encode_plus(
    sent_batch_1,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=args.max_seq_length,
)
token_batch_2 = tokenizer.batch_encode_plus(
    sent_batch_2,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=args.max_seq_length,
)

token_batch_1 = token_batch_1.to(accelerator.device)
token_batch_2 = token_batch_2.to(accelerator.device)

batch_emb_1 = model(**token_batch_1, output_hidden_states=True, return_dict=False, sent_emb=True)[1]         
batch_emb_2 = model(**token_batch_2, output_hidden_states=True, return_dict=False, sent_emb=True)[1]         

rel_embs = model.rel_emb.embedding.weight.detach()


cos_sim = nn.CosineSimilarity(dim=-1)

rel_scores = []
for rel_emb in rel_embs:
    batch_emb_1_trans = batch_emb_1 + rel_emb

    score = cos_sim(batch_emb_1_trans, batch_emb_2)
    score = score.cpu().detach().numpy()
    rel_scores.append(score)

rel_scores = np.array(rel_scores).transpose()


# Display all relation scores
for i in range(len(sent_batch_1)):
    table = PrettyTable()
    table.field_names = ['Relationship', 'Score']

    print("")
    print("Sentence 1: " + sent_batch_1[i])
    print("Sentence 2: " + sent_batch_2[i])
    print("")

    for j in range(args.num_rel):
        table.add_row([index2rel[j], rel_scores[i,j]])

    print(table)


