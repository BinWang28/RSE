#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data_a12/binwang/research/RSE/rse_src/SentEval/senteval/sim_func.py
# Project: /data_a12/binwang/research/RSE/rse_src/SentEval/senteval
# Created Date: 1970-01-01 07:30:00
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----

import torch
import numpy as np

def sim_max_func(emb1, emb2, rel_embs, cosine_sim):
    """ Compute similarity between embeddings with relation modeling with maximum function """
    comb_scores = []
    for single_rel_emb in rel_embs:
        score = cosine_sim(emb1 + single_rel_emb, emb2)
        comb_scores.append(score)
        score = cosine_sim(emb2 + single_rel_emb, emb1)
        comb_scores.append(score)
    return max(comb_scores)


def sim_mean_func(emb1, emb2, rel_embs, cosine_sim):
    """ Compute similarity between embeddings with relation modeling with mean function"""
    comb_scores = []
    for single_rel_emb in rel_embs:
        score = cosine_sim(emb1 + single_rel_emb, emb2)
        comb_scores.append(score)
        score = cosine_sim(emb2 + single_rel_emb, emb1)
        comb_scores.append(score)
    return sum(comb_scores) / len(comb_scores)


def sim_rel_func(emb1, emb2, rel_embs, cosine_sim, rel_weights):
    """ Similarity based on one relation embedding matrix """

    # L2 distance
    # value1 = 15 - torch.norm(emb1+single_rel_emb-emb2, p=2.0)
    # value2 = 15 - torch.norm(emb1+single_rel_emb-emb2, p=2.0)
    # return max(value1, value2)

    # Cosine Similarity
    #return max(cosine_sim(emb1 + single_rel_emb, emb2), cosine_sim(emb2 + single_rel_emb, emb1))
    rel_weights = np.array([float(val) for val in rel_weights])
    rel_weights = rel_weights / sum(rel_weights)
    
    scores1 = []
    for single_rel_emb in rel_embs:
        score = cosine_sim(emb1 + single_rel_emb, emb2)
        scores1.append(score)
    score1 = rel_weights.dot(scores1)

    scores2 = []
    for single_rel_emb in rel_embs:
        score = cosine_sim(emb2 + single_rel_emb, emb1)
        scores2.append(score)
    score2 = rel_weights.dot(scores2)

    return max(score1, score2)


def sim_weight_func(emb1, emb2, rel_embs, cosine_sim, rel_weights):
    """ Compute similarity between embeddings with relation modeling with  weights """
        
    comb_scores = []
    for single_rel_emb in rel_embs:
        score = cosine_sim(emb1 + single_rel_emb, emb2)
        comb_scores.append(score)
    score1 = sum(comb_scores * rel_weights)
    
    comb_scores = []
    for single_rel_emb in rel_embs:
        score = cosine_sim(emb2 + single_rel_emb, emb1)
        comb_scores.append(score)
    score2 = sum(comb_scores * rel_weights)

    return max(score1,score2)


