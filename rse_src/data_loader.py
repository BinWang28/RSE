#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data07/binwang/research/RSE/src/data_loader.py
# Project: /data07/binwang/research/RSE/src
# Created Date: 2022-07-24 16:12:19
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----

from collections import defaultdict
import json
import random
import logging

from tqdm import tqdm

import datasets


def raw_data_loader(args):
    """ load raw dataset """

    train_dict, rel2index, index2rel = load_from_json(args, args.train_files)
    raw_datasets = datasets.DatasetDict({"train":train_dict})

    return raw_datasets, rel2index, index2rel


def load_from_json(args, file_path):
    """ load from json and convert to DATASET """
    logging.info("Load data from json files: \n {} \n".format(file_path))

    # Save: index2rel and rel2index
    index2rel = args.rel_types
    rel2index = {}
    for index, value in enumerate(index2rel):
        rel2index[value] = index

    rel_num_count = {value: 0 for value in rel2index.keys()}

    # Read all datasets - to find negative samples
    rel2sent_pairs = {}
    for filename in file_path:
        with open(filename, 'r') as f:
            all_samples = json.load(f)

        for sample in all_samples:

            sent1 = sample['sentence1']
            sent2 = sample['sentence2']
            label = sample['label']
            
            if label not in rel2sent_pairs:
                rel2sent_pairs[label] = {}
            if sent1 not in rel2sent_pairs[label]:
                rel2sent_pairs[label][sent1] = []

            rel2sent_pairs[label][sent1].append(sent2)

    # Dataset format
    sentence1_list = []
    sentence2_list = []
    label_list     = []

    if args.add_hard_neg:
        sentence3_list = []

    for label, pairs in rel2sent_pairs.items():
        
        # Exclude non-specified relations
        if label not in rel2index:
            continue

        all_sents_rel = list(rel2sent_pairs[label].values())
        all_sents_rel = [sent for sub_g in all_sents_rel for sent in sub_g]

        max_sample_cur_relation = args.rel_max_samples[rel2index[label]]
        if max_sample_cur_relation == -1: max_sample_cur_relation = 1e10

        for sent1, sent2_group in tqdm(pairs.items()):
            for sent2 in sent2_group:

                # Skip if reaches the maximum bound
                if rel_num_count[label] >= max_sample_cur_relation:
                    continue
                rel_num_count[label] += 1

                if args.add_hard_neg:
                    # Find the negative samples
                    # Special treatment for Entialment and Contradiction
                    if label == 'entailment':
                        contradiction_pairs = rel2sent_pairs['contradiction']
                        if sent1 in contradiction_pairs:
                            sent3 = random.choice(contradiction_pairs[sent1])
                        else:
                            contra_sents = contradiction_pairs.values()
                            contra_sents = [sent for sents_set in contra_sents for sent in sents_set]
                            sent3        = random.choice(contra_sents)

                    elif label == 'contradiction':
                        entailment_pairs = rel2sent_pairs['entailment']
                        if sent1 in entailment_pairs:
                            sent3 = random.choice(entailment_pairs[sent1])
                        else:
                            entail_sents = entailment_pairs.values()
                            entail_sents = [sent for sents_set in entail_sents for sent in sents_set]
                            sent3        = random.choice(entail_sents)

                    else:
                        # Random select one from the same relation
                        sent3 = random.choice(all_sents_rel)

                # Add to training
                sentence1_list.append(sent1)
                sentence2_list.append(sent2)
                label_list.append(rel2index[label])

                if args.add_hard_neg: 
                    sentence3_list.append(sent3)

    # Contruct the dataset
    id_list = list(range(0, len(sentence1_list)))

    logging.info("Relation to index: \n{} \n".format(rel2index))
    logging.info("Relation and number of samples: \n {} \n".format(rel_num_count))
    args.num_rel = len(rel2index)

    if args.add_hard_neg:
        data_dict = {
                    'id'       : id_list,
                    'sentence1': sentence1_list,
                    'sentence2': sentence2_list,
                    'sentence3': sentence3_list,
                    'labels'   : label_list
                    }
    else:
        data_dict = {
                    'id'       : id_list,
                    'sentence1': sentence1_list,
                    'sentence2': sentence2_list,
                    'labels'   : label_list
                    }

    data_dict = datasets.Dataset.from_dict(data_dict)
    logging.info("Sample from loaded data: " + str(data_dict[0]))

    return data_dict, rel2index, index2rel



