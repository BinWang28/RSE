#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data_a12/binwang/research/RSE/rse_src_debug/loss.py
# Project: /data_a12/binwang/research/RSE/rse_src_debug
# Created Date: 1970-01-01 07:30:00
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----


import torch
import torch.nn as nn



class Similarity(nn.Module):    
    ''' Cosine Similarity with temperature '''
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
    
    def forward(self, x, y):
        x, y = x.unsqueeze(1), y.unsqueeze(0)
        # Cosine similarity
        return self.cos(x, y) / self.temp



class ContrastiveLoss:
    def __init__(self, model, temperature):
        self.model = model # may not be necessary to load the model
        self.sim   = Similarity(temp=temperature)

    def __call__(self, representations):

        # Split representations
        num_sent = representations.shape[1]
        if num_sent == 2:
            sent1_rep, sent2_rep = representations[:,0], representations[:,1]
        elif num_sent == 3:
            sent1_rep, sent2_rep, sent3_rep = representations[:,0], representations[:,1], representations[:,2]

        sent1_sent2_cos_sim = self.sim(sent1_rep, sent2_rep)
        full_cos_sim    = torch.cat([sent1_sent2_cos_sim], 1)

        if num_sent == 3:
            sent1_sent3_cos_sim = self.sim(sent1_rep, sent3_rep)
            full_cos_sim    = torch.cat([full_cos_sim, sent1_sent3_cos_sim], 1)

        labels   = torch.arange(full_cos_sim.size(0)).long().to(self.model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss     = loss_fct(full_cos_sim, labels)

        return loss


