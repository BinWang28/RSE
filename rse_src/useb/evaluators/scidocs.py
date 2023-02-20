from .base import BaseEvaluator
import re
import os
import argparse
import tqdm
from typing import Dict, List, Set
from sklearn.metrics import average_precision_score, ndcg_score
from scipy.stats import pearsonr, spearmanr
import logging
import numpy as np
import pickle
import json
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
import pytrec_eval
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')


class SciDocsDataset(object):

    def __init__(self, datasets_dir):
        dnames = ['cite', 'cocite', 'coview', 'coread']
        self.dnames = dnames
        assert 'data.json' in os.listdir(datasets_dir)
        fdata = os.path.join(datasets_dir, 'data.json')
        with open(fdata, 'r') as f:
            self.data = json.load(f)


class SciDocsEvaluator(BaseEvaluator, SciDocsDataset):
    name = 'scidocs'
    main_metric = 'map_scidocs_cosine_avg'

    def __init__(self, semb_fn, datasets_dir=None, bsz=32, show=True):
        BaseEvaluator.__init__(self, semb_fn, bsz, show)
        SciDocsDataset.__init__(self, datasets_dir)

    @property
    def metric_names(self):
        mnames = []
        for dname in self.dnames:
            for metric in ['map', 'ndcg']:
                for distance in ['euclidean', 'cosine']:
                    mnames.append(f'{metric}_{dname}_{distance}')
        for metric in ['map', 'ndcg']:
            for distance in ['euclidean', 'cosine']:
                mnames.append(f'{metric}_{distance}_avg')
        return mnames

    def _get_sent(self, pid):
        corpus = self.data['corpus']
        if pid not in corpus:
            return None
        title = corpus[pid]['title']
        if title is None:
            title = ''
        return title

    def _run(self, eval_type):
        qrels: Dict[str, Dict[str, Dict[str, int]]] = self.data[eval_type]
        results = {}
        show = bool(self.show)

        cos_sim = nn.CosineSimilarity(dim=-1)

        for dname, qrel in qrels.items():
            run_euclidean = {}
            run_cosine = {}
            for qid, doc_dict in tqdm.tqdm(qrel.items(), disable=not self.show):
                query_text = self._get_sent(qid)    
                if not query_text:
                    continue
                dids = [did for did in doc_dict if self._get_sent(did)]
                doc_texts = map(lambda did: self._get_sent(did), dids)
                batch_texts = [query_text]
                batch_texts.extend(doc_texts)
                self.show = False  # to mute the progress bar of the next line
                embs: torch.Tensor = self._text2se(batch_texts, normalize=False)
                qemb = embs[:1]  # (1, hdim)
                dembs = embs[1:]

                rel_embs, rel_weights = self.semb_fn(None, out_rel_emb=True)
                rel_weights = np.array([float(weight) for weight in rel_weights])
                rel_weights = rel_weights / sum(rel_weights)

                assert rel_embs.shape[0] == len(rel_weights), "Number of relation embedding and weights must match."

                scores_rel = []
                for rel_emb in rel_embs:
                    qemb_s = qemb + rel_emb
                    score = cos_sim(qemb_s, dembs)
                    scores_rel.append(score.numpy())
                scores_rel = np.array(scores_rel)

                scores_cosine = rel_weights.dot(scores_rel).tolist()
                run_cosine[qid] = dict(zip(dids, scores_cosine))

            self.show = bool(show)
            pytrec_evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg'})
                        
            result_cosine: Dict[str, Dict[str, float]] = pytrec_evaluator.evaluate(run_cosine)
            results[f'map_scidocs_{dname}_cosine'] = np.mean([items['map'] for items in result_cosine.values()])
            results[f'ndcg_scidocs_{dname}_cosine'] = np.mean([items['ndcg'] for items in result_cosine.values()])
        
        results[f'map_scidocs_cosine_avg'] = np.mean([results[f'map_scidocs_{dname}_cosine'] for dname in qrels])
        results[f'ndcg_scidocs_cosine_avg'] = np.mean([results[f'ndcg_scidocs_{dname}_cosine'] for dname in qrels])
        return results