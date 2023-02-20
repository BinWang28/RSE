# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Relation prediction on sentence pairs
'''
from __future__ import absolute_import, division, unicode_literals

import os
import sys

import json
import logging

import torch
import numpy as np

from tqdm import tqdm

from senteval.utils import cosine


class RelPredEval(object):
    def __init__(self, task_path, seed=1111):
        logging.info('***** Transfer task: Sentence Relation Prediction - Retrieval *****')

        # Get files and data
        self.seed = seed
        dev, test = self.loadFile(task_path)
        self.relp_data = {'dev': dev, 'test': test}

    def do_prepare(self, params, prepare):
        samples = self.relp_data['dev']['sents'] + self.relp_data['test']['sents']
        prepare(params, samples)
        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

    def loadFile(self, fpath):

        relp_data = {}
        for split in ['validation', 'test']:
            list_pairs = []
            list_sents = []

            with open(os.path.join(fpath, split + '.json'), 'r') as f:
                all_samples = json.load(f)

            for sample in all_samples:
                if sample['sentence1'] not in list_sents:
                    list_sents.append(sample['sentence1'])
                if sample['sentence2'] not in list_sents:
                    list_sents.append(sample['sentence2'])
                list_pairs.append(sample)

            relp_data[split] = {'pairs': list_pairs, 'sents': list_sents}

        return relp_data['validation'], relp_data['test']

    def run(self, params, batcher):

        rel2index = params['rel2index']

        results = {}
        for key in self.relp_data:
            set_data   = self.relp_data[key]
            list_sents = set_data['sents']
            list_pairs = set_data['pairs']

            # Retrieve the embedding of all candidate sentences
            nsent    = len(list_sents)
            all_embs = []
            for ii in range(0, nsent, params.batch_size):
                batch_sents = list_sents[ii : ii + params.batch_size]
                batch_sents = [sent.split(' ') for sent in batch_sents]
                embeddings, rel_emb  = batcher(params, batch_sents, rel_emb_out=True)
                all_embs.append(embeddings)

            all_embs   = torch.cat(all_embs)
            all_embs_n = torch.nn.functional.normalize(all_embs, dim=1)

            # Compute retrieval performance for each sample set
            all_ranks = {'all': []}
            for sample in list_pairs:

                rel_label = sample['label']
                if rel_label not in rel2index:
                    continue

                sent1       = sample['sentence1']
                sent2       = sample['sentence2']
                sent1_index = list_sents.index(sent1)
                sent2_index = list_sents.index(sent2)

                sent1_emb   = all_embs[sent1_index]
                cur_rel_emb = rel_emb[rel2index[rel_label]]

                rel_sent1_emb = torch.nn.functional.normalize((sent1_emb + cur_rel_emb).unsqueeze(0)).squeeze()

                all_scores = all_embs_n.matmul(rel_sent1_emb)
                pair_score = all_scores[sent2_index]

                # remove sent1
                if sent1 != sent2:
                    all_scores = torch.cat([all_scores[0:sent1_index], all_scores[sent1_index+1:]])

                all_scores = torch.sort(all_scores, descending=True).values
                cur_rank   = all_scores.tolist().index(pair_score.tolist()) + 1

                if rel_label not in all_ranks:
                    all_ranks[rel_label] = [cur_rank]
                else:
                    all_ranks[rel_label].append(cur_rank)

                all_ranks['all'].append(cur_rank)

            # Compute metrics - MR, MRR, HITS1, HITS3, HITS10
            cur_key_res = {}
            for category in all_ranks:
                res         = all_ranks[category]
                res         = np.array(res)
                num_samples = len(res)

                MR_res     = np.mean(res)
                MRR_res    = np.mean(1.0/res)
                Hits1_res  = np.sum(res <= 1) / num_samples
                Hits3_res  = np.sum(res <= 3) / num_samples
                Hits10_res = np.sum(res <= 10) / num_samples

                cur_key_res[category] = {
                    'MR'    : MR_res,
                    'MRR'   : MRR_res,
                    'Hits1' : Hits1_res,
                    'Hits3' : Hits3_res,
                    'Hits10': Hits10_res
                }

            results[key] = cur_key_res
        
        return results
