#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data07/binwang/research/RSE/src/eval.py
# Project: /data07/binwang/research/RSE/src
# Created Date: 2022-07-25 09:28:53
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----

import logging
import sys
import json
import torch

import numpy as np

import scipy.spatial as sp

from utils import format_output, format_output_useb

# =  =  = Import SentEval =  =  =  =  = 
PATH_TO_SENTEVAL = './rse_src/SentEval'
PATH_TO_DATA     = './rse_src/SentEval/data'
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# =  =  = Import useb =  =  =  =  = 
from useb import run as useb_run


def excute_eval(args, logger, accelerator, model, tokenizer, eval_task, eval_mode='fast'):
    """ Perform Evaluation based on input tasks. """

    def prepare(params, samples):
        return

    def batcher(params, batch, rel_emb_out=False):

        sentences = [' '.join(item) for item in batch]

        token_batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        )

        token_batch = token_batch.to(accelerator.device)
        with torch.no_grad():
            outputs = model(**token_batch, output_hidden_states=True, return_dict=False, sent_emb=True)            
            pooler_output = outputs[1]

        if args.layer_aggregation:
            last_three_feature = torch.stack(outputs[2][-args.layer_aggregation:])[:,:,0,:]
            last_four_feature  = torch.cat([last_three_feature, pooler_output.unsqueeze(0)])
            pooler_output      = torch.mean(last_four_feature, axis=0)
        
        if rel_emb_out:
            return pooler_output.cpu(), rel_emb
        else:
            return pooler_output.cpu()


    # SentEval Parameters - full model training for better performance
    if eval_mode == 'fast':
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                             'tenacity': 3, 'epoch_size': 2}
    elif eval_mode == 'full':
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                            'tenacity': 5, 'epoch_size': 4}
    else:
        NotImplementedError
    
    # Print relation similarity
    if args.mode == "RSE":
        rel_emb = model.rel_emb.embedding.weight.detach().cpu().numpy()
        rel_emb_norm = np.linalg.norm(rel_emb, axis=1)
        logging.info("\nRelation Similarity: \n{}".format(1-sp.distance.cdist(rel_emb, rel_emb, 'cosine')))
        logging.info("\nRelation Norm: \n{}".format(rel_emb_norm))
    
        params['sim_func']  = args.sim_func
        params['rel2index'] = args.rel2index

    # Execute evaluation
    se = senteval.engine.SE(params, batcher, prepare)
    res_metrics = se.eval(eval_task)

    # Format the output, only keep results of interest
    display_metrics = format_output(res_metrics)
    logger.info("\n" + str(display_metrics))

    return res_metrics, display_metrics
    

def eval_on_useb(args, logger, accelerator, model, tokenizer, sim_weights, mode='test'):
    ''' Evaluation on useb tasks '''

    def semb_fn(sentences, out_rel_emb=False):
        ''' Return only sentence embedding is out_rel_emb=False '''
        ''' Return only relation embedding is out_rel_emb=True '''

        if not out_rel_emb:
            token_batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=args.max_seq_length,
            )

            token_batch = token_batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**token_batch, output_hidden_states=True, return_dict=False, sent_emb=True)            
                pooler_output = outputs[1]

            return pooler_output.cpu()

        elif out_rel_emb:
            rel_emb = model.rel_emb.embedding.weight.detach().cpu().numpy()
            return rel_emb, sim_weights

        else:
            NotImplementedError

    useb_res, useb_res_main = useb_run(
        semb_fn_askubuntu   = semb_fn,
        semb_fn_cqadupstack = semb_fn,
        semb_fn_twitterpara = semb_fn,
        semb_fn_scidocs     = semb_fn,
        eval_type           = mode,
        data_eval_path      = './rse_src/useb/data/data-eval'
    )

    # Format the output, only keep results of interest
    useb_diaplay_metrics = format_output_useb(useb_res)
    logger.info("\n" + str(useb_diaplay_metrics))

    return useb_res, useb_res_main, useb_diaplay_metrics


def eval_on_evalrank(args, logger, accelerator, model, tokenizer):
    ''' Evaluation on EvalRank tasks https://aclanthology.org/2022.acl-long.419/ '''

    # Load Data

    pos_pairs = []
    all_sents = []

    logger.info('*** Prepare pos sentence pairs for ranking evaluation ***')

    with open('rse_src/evalrank/' + 'pos_pair.txt', 'r') as f: 
        lines = f.readlines()
        for line in lines:
            sent1, sent2 = line.strip().split('\t')
            pos_pairs.append([sent1, sent2])
    
    
    logger.info("Loading Background Sentences for Ranking")

    for item in pos_pairs:
        if item[0] not in all_sents: all_sents.append(item[0])
        if item[1] not in all_sents: all_sents.append(item[1])

    with open('rse_src/evalrank/' + 'background_sent.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line not in all_sents:
                all_sents.append(line)

    logger.info('{} sentences as background sentences'.format(len(all_sents)))

    import pdb; pdb.set_trace()
