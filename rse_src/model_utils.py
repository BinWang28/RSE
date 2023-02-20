#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data07/binwang/research/RSE/src/model.py
# Project: /data07/binwang/research/RSE/src
# Created Date: 2022-07-25 09:28:03
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----


from turtle import forward
import torch
import torch.nn as nn

from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


class Pooler(nn.Module):
    '''
    Parameter-free poolers to get the sentence embedding.
    'CLS' the first token representation
    '''

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls"], "unsupported pooling type {}".format(self.pooler_type)

    def forward(self, attention_mask, outputs):
        last_hidden   = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type == 'cls':
            return last_hidden[:, 0]
        else:
            NotImplementedError


class MLPLayer(nn.Module):
    ''' MLP layer upon BERT/RoBERTa encoder '''
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


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


class EmbeddingLayer(nn.Module):
    ''' relation embedding layer '''
    def __init__(self, config, num_rel):
        super().__init__()
        self.embedding = nn.Embedding(num_rel, config.hidden_size)

    def forward(self, indexes, **kwargs):
        return self.embedding(indexes.squeeze())


def RSE_init(self, config):
    ''' RSE init '''
    self.pooler_type = self.args.pooler_type
    self.pooler      = Pooler(self.pooler_type)

    # add additional mlp if using 'cls'
    if self.pooler_type == "cls":
        self.mlp = MLPLayer(config)


    if self.args.mode == 'RSE':
        self.rel_emb     = EmbeddingLayer(config, self.args.num_rel)

        # Need to have a good initialization if learning rate for relational embedding is small
        # nn.init.normal_(self.rel_emb.embedding.weight, 0.0, 0.4)

    # Another variant is to initialize before relational embedding matrix
    self.init_weights()


def rse_forward(self,
    encoder,
    input_ids            = None,
    attention_mask       = None,
    token_type_ids       = None,
    position_ids         = None,
    head_mask            = None,
    inputs_embeds        = None,
    labels               = None,
    output_attentions    = None,
    output_hidden_states = None,
    return_dict          = None
    ):
    """ relational sentence embedding forward pass """

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    batch_size = input_ids.size(0)
    num_sent   = input_ids.size(1)

    # Flatten input for encoding
    input_ids      = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask       = attention_mask,
        token_type_ids       = token_type_ids,
        position_ids         = position_ids,
        head_mask            = head_mask,
        inputs_embeds        = inputs_embeds,
        output_attentions    = output_attentions,
        output_hidden_states = False,
        return_dict          = True,
    )

    pooler_output = self.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    if self.pooler_type == "cls":
        pooler_output = self.mlp(pooler_output)

    # Separate representations
    sr1, sr2 = pooler_output[:,0], pooler_output[:,1]
    if num_sent == 3:
        sr3 = pooler_output[:,2]

    # Rel - Sim
    if self.args.mode == "RSE":
        batch_rel_embs = self.rel_emb(labels)
        sr1 = sr1 + batch_rel_embs

    # Prepare representation
    output = torch.stack([sr1, sr2], axis=1)
    if num_sent == 3:
        output = torch.stack([sr1, sr2, sr3], axis=1)

    return output


def sentemb_forward(
    self,
    encoder,
    input_ids            = None,
    attention_mask       = None,
    token_type_ids       = None,
    position_ids         = None,
    head_mask            = None,
    inputs_embeds        = None,
    labels               = None,
    output_attentions    = None,
    output_hidden_states = None,
    return_dict          = None,
    ):
    ''' only obtain sentence embedding after training '''
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask       = attention_mask,
        token_type_ids       = token_type_ids,
        position_ids         = position_ids,
        head_mask            = head_mask,
        inputs_embeds        = inputs_embeds,
        output_attentions    = output_attentions,
        output_hidden_states = output_hidden_states,
        return_dict          = True,
    )

    pooler_output = self.pooler(attention_mask, outputs)
    
    if self.pooler_type == "cls" and not self.args.mlp_only_train:
        pooler_output = self.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[1:]
    else:
        NotImplementedError
