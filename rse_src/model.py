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

from transformers.modeling_outputs import SequenceClassifierOutput

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

from model_utils import RSE_init, rse_forward, sentemb_forward


class BertForRSE(BertPreTrainedModel):
    ''' prepare BERT relational sentence embedding model '''

    def __init__(self, config, *model_args, **kwargs):
        super().__init__(config)
        
        self.args = kwargs["args"]
        self.bert = BertModel(config, add_pooling_layer=False)
        
        if self.args.add_neg:
            self.save_sent_fea = None

        # additional modules init
        RSE_init(self, config)

    def forward(
        self,
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
        sent_emb             = False,
    ):

        if sent_emb:
            # only obtain embedding for evaluation
            return sentemb_forward(
                self,
                self.bert,
                input_ids            = input_ids,
                attention_mask       = attention_mask,
                token_type_ids       = token_type_ids,
                position_ids         = position_ids,
                head_mask            = head_mask,
                inputs_embeds        = inputs_embeds,
                labels               = labels,
                output_attentions    = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict          = return_dict,
                )
        else:
            # full forward pass for training
            return rse_forward(
                self,
                self.bert,
                input_ids            = input_ids,
                attention_mask       = attention_mask,
                token_type_ids       = token_type_ids,
                position_ids         = position_ids,
                head_mask            = head_mask,
                inputs_embeds        = inputs_embeds,
                labels               = labels,
                output_attentions    = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict          = return_dict,
                )



class RoBERTaForRSE(RobertaPreTrainedModel):
    ''' prepare BERT relational sentence embedding model '''

    def __init__(self, config, *model_args, **kwargs):
        super().__init__(config)
        
        self.args    = kwargs["args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # additional modules init
        RSE_init(self, config)

    def forward(
        self,
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
        sent_emb             = False,
    ):

        if sent_emb:
            # only obtain embedding for evaluation
            return sentemb_forward(
                self,
                self.roberta,
                input_ids            = input_ids,
                attention_mask       = attention_mask,
                token_type_ids       = token_type_ids,
                position_ids         = position_ids,
                head_mask            = head_mask,
                inputs_embeds        = inputs_embeds,
                labels               = labels,
                output_attentions    = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict          = return_dict,
                )
        else:
            # full forward pass for training
            return rse_forward(
                self,
                self.roberta,
                input_ids            = input_ids,
                attention_mask       = attention_mask,
                token_type_ids       = token_type_ids,
                position_ids         = position_ids,
                head_mask            = head_mask,
                inputs_embeds        = inputs_embeds,
                labels               = labels,
                output_attentions    = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict          = return_dict,
                )
