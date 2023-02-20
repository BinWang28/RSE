#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data07/binwang/research/RSE/src/utils.py
# Project: /data07/binwang/research/RSE/src
# Created Date: 2022-07-24 16:42:50
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----

import logging
from prettytable import PrettyTable



def format_output(metrics):
    """ format the output for printing """

    table = PrettyTable()
    table.title = 'Eval Results'
    table.field_names = ['Task', 'Value 1', 'Value 2']

    if 'STS12' in metrics:
        try:
            sts12_res = metrics['STS12']['all']['spearman']['all']
            table.add_row(['STS12', '- - -', '(all) {:.6f}'.format(sts12_res)])
            sts13_res = metrics['STS13']['all']['spearman']['all']
            table.add_row(['STS13', '- - -', '(all) {:.6f}'.format(sts13_res)])
            sts14_res = metrics['STS14']['all']['spearman']['all']
            table.add_row(['STS14', '- - -', '(all) {:.6f}'.format(sts14_res)])
            sts15_res = metrics['STS15']['all']['spearman']['all']
            table.add_row(['STS15', '- - -', '(all) {:.6f}'.format(sts15_res)])
            sts16_res = metrics['STS16']['all']['spearman']['all']
            table.add_row(['STS16', '- - -', '(all) {:.6f}'.format(sts16_res)])
        except:
            raise Exception('all STS12-16 must be evaluated simultaneously')

    if 'STSBenchmark_unsup' in metrics:
        dev_res = metrics['STSBenchmark_unsup']['dev']['spearman'][0]
        stsb_u_res = metrics['STSBenchmark_unsup']['all']['spearman']['all']
        table.add_row(['STS-B Unsup', '(dev) {:.6f}'.format(dev_res), '(all) {:.6f}'.format(stsb_u_res)])

    if 'SICKRelatedness_unsup' in metrics:
        sickr_u_res = metrics['SICKRelatedness_unsup']['all']['spearman']['all']
        table.add_row(['SICK-R Unsup', '- - -', '(all) {:.6f}'.format(sickr_u_res)])

    if 'STS12' in metrics and 'STSBenchmark_unsup' in metrics and 'SICKRelatedness_unsup' in metrics:
        # Report AVG.
        avg_res = (sts12_res + sts13_res + sts14_res + sts15_res + sts16_res + stsb_u_res + sickr_u_res) / 7
        table.add_row(['STS AVG', '- - -', '{:.6f}'.format(avg_res)])


    if 'STR' in metrics:
        str_res = metrics['STR']['test']['spearman'][0]
        table.add_row(['---------------', '---------------', '---------------'])
        table.add_row(['STR', '- - -', '(all) {:.6f}'.format(str_res)])


    # =  =  =  Downstream =  =  =
    if any([True for item in metrics.keys() if item in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']]):
        table.add_row(['---------------', '---------------', '---------------'])

    if 'MR' in metrics:
        mr_res = metrics['MR']['acc']
        table.add_row(['MR', '- - -', '{:.2f}'.format(mr_res)])
    if 'CR' in metrics:
        cr_res = metrics['CR']['acc']
        table.add_row(['CR', '- - -', '{:.2f}'.format(cr_res)])
    if 'SUBJ' in metrics:
        subj_res = metrics['SUBJ']['acc']
        table.add_row(['SUBJ', '- - -', '{:.2f}'.format(subj_res)])
    if 'MPQA' in metrics:
        mpqa_res = metrics['MPQA']['acc']
        table.add_row(['MPQA', '- - -', '{:.2f}'.format(mpqa_res)])
    if 'SST2' in metrics:
        sst2_res = metrics['SST2']['acc']
        table.add_row(['SST2', '- - -', '{:.2f}'.format(sst2_res)])
    if 'TREC' in metrics:
        trec_res = metrics['TREC']['acc']
        table.add_row(['TREC', '- - -', '{:.2f}'.format(trec_res)])
    if 'MRPC' in metrics:
        mrpc_res = metrics['MRPC']['acc']
        table.add_row(['MRPC', '- - -', '{:.2f}'.format(mrpc_res)])

    try:
        downstream_avg = (mr_res + cr_res + subj_res + mpqa_res + sst2_res + trec_res + mrpc_res) / 7
        table.add_row(['Downstream AVG', '- - -', '{:.2f}'.format(downstream_avg)])
    except:
        pass


    # =  =  =  Sup SICK, STSB =  =  =
    if any([True for item in metrics.keys() if item in ['STSBenchmark', 'SICKRelatedness', 'SICKEntailment']]):
        table.add_row(['---------------', '---------------', '---------------'])

    if 'STSBenchmark' in metrics:
        stsb_res = metrics['STSBenchmark']['spearman']
        table.add_row(['STS-B Sup', '- - -', '{:.2f}'.format(stsb_res)])
    if 'SICKRelatedness' in metrics:
        sickr_res = metrics['SICKRelatedness']['spearman']
        table.add_row(['SICK-R Sup', '- - -', '{:.2f}'.format(sickr_res)])
    if 'SICKEntailment' in metrics:
        sicke_res = metrics['SICKEntailment']['acc']
        table.add_row(['SICK-E', '- - -', '{:.2f}'.format(sicke_res)])


    # =  =  = Relation Prediction =  =  =
    if 'RelPred' in metrics:
        dev_res = metrics['RelPred']['dev']
        for key, value in dev_res.items():
            table.add_row(['---------------', '---------------', '---------------'])
            table.add_row(['RelPred - dev - {}'.format(key), 'MR', '{:.2f}'.format(value['MR'])])
            table.add_row(['RelPred - dev - {}'.format(key), 'MRR', '{:.2f}'.format(value['MRR'])])
            table.add_row(['RelPred - dev - {}'.format(key), 'Hits1', '{:.2f}'.format(value['Hits1'])])
            table.add_row(['RelPred - dev - {}'.format(key), 'Hits3', '{:.2f}'.format(value['Hits3'])])
            table.add_row(['RelPred - dev - {}'.format(key), 'Hits10', '{:.2f}'.format(value['Hits10'])])

        test_res = metrics['RelPred']['test']
        for key, value in test_res.items():
            table.add_row(['---------------', '---------------', '---------------'])
            table.add_row(['RelPred - test - {}'.format(key), 'MR', '{:.2f}'.format(value['MR'])])
            table.add_row(['RelPred - test - {}'.format(key), 'MRR', '{:.2f}'.format(value['MRR'])])
            table.add_row(['RelPred - test - {}'.format(key), 'Hits1', '{:.2f}'.format(value['Hits1'])])
            table.add_row(['RelPred - test - {}'.format(key), 'Hits3', '{:.2f}'.format(value['Hits3'])])
            table.add_row(['RelPred - test - {}'.format(key), 'Hits10', '{:.2f}'.format(value['Hits10'])])


    # =  =  =  Probing =  =  =
    if any([True for item in metrics.keys() if item in [
            'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift',
            'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']]):
        table.add_row(['---------------', '---------------', '---------------'])

    if 'Length' in metrics:
        length_res = metrics['Length']['acc']
        table.add_row(['Length', '- - -', '{:.2f}'.format(length_res)])
    if 'WordContent' in metrics:
        wc_res = metrics['WordContent']['acc']
        table.add_row(['WordContent', '- - -', '{:.2f}'.format(wc_res)])
    if 'Depth' in metrics:
        depth_res = metrics['Depth']['acc']
        table.add_row(['Depth', '- - -', '{:.2f}'.format(depth_res)])
    if 'TopConstituents' in metrics:
        tc_res = metrics['TopConstituents']['acc']
        table.add_row(['TopConstituents', '- - -', '{:.2f}'.format(tc_res)])
    if 'BigramShift' in metrics:
        bs_res = metrics['BigramShift']['acc']
        table.add_row(['BigramShift', '- - -', '{:.2f}'.format(bs_res)])
    if 'Tense' in metrics:
        tense_res = metrics['Tense']['acc']
        table.add_row(['Tense', '- - -', '{:.2f}'.format(tense_res)])
    if 'SubjNumber' in metrics:
        sjn_res = metrics['SubjNumber']['acc']
        table.add_row(['SubjNumber', '- - -', '{:.2f}'.format(sjn_res)])
    if 'ObjNumber' in metrics:
        obn_res = metrics['ObjNumber']['acc']
        table.add_row(['ObjNumber', '- - -', '{:.2f}'.format(obn_res)])
    if 'OddManOut' in metrics:
        omo_res = metrics['OddManOut']['acc']
        table.add_row(['OddManOut', '- - -', '{:.2f}'.format(omo_res)])
    if 'CoordinationInversion' in metrics:
        cooi_res = metrics['CoordinationInversion']['acc']
        table.add_row(['CoordinationInversion', '- - -', '{:.2f}'.format(cooi_res)])

    return table




def format_output_useb(metrics):
    """ format the output for printing """

    table = PrettyTable()
    table.title = 'Eval Results'
    table.field_names = ['Task', 'Value 1', 'Value 2']

    table.add_row(['AskU.', 'MAP', '{:.2f}'.format(metrics['askubuntu']['map_askubuntu_title'])])
    table.add_row(['---------------', '---------------', '---------------'])

    table.add_row(['CQADup.', 'MAP', '{:.2f}'.format(metrics['cqadupstack']['map@100_cqadupstack_avg'])])
    table.add_row(['---------------', '---------------', '---------------'])

    table.add_row(['TwitterP.', 'TURL', '{:.2f}'.format(metrics['twitterpara']['ap_twitter_twitterurl'])])
    table.add_row(['TwitterP.', 'PIT', '{:.2f}'.format(metrics['twitterpara']['ap_twitter_pit'])])
    table.add_row(['TwitterP.', 'Avg.', '{:.2f}'.format(metrics['twitterpara']['ap_twitter_avg'])])
    table.add_row(['---------------', '---------------', '---------------'])

    table.add_row(['SciDocs', 'Cite', '{:.2f}'.format(metrics['scidocs']['map_scidocs_cite_cosine'])])
    table.add_row(['SciDocs', 'CC', '{:.2f}'.format(metrics['scidocs']['map_scidocs_cocite_cosine'])])
    table.add_row(['SciDocs', 'CR', '{:.2f}'.format(metrics['scidocs']['map_scidocs_coread_cosine'])])
    table.add_row(['SciDocs', 'CV', '{:.2f}'.format(metrics['scidocs']['map_scidocs_coview_cosine'])])
    table.add_row(['SciDocs', 'Avg.', '{:.2f}'.format(metrics['scidocs']['map_scidocs_cosine_avg'])])
    table.add_row(['---------------', '---------------', '---------------'])

    overall_avg = ( metrics['askubuntu']['map_askubuntu_title'] + 
                    metrics['cqadupstack']['map@100_cqadupstack_avg'] +
                    metrics['twitterpara']['ap_twitter_avg'] +
                    metrics['scidocs']['map_scidocs_cosine_avg']
                    ) / 4.0

    table.add_row(['USEB_all', 'Avg.', '{:.2f}'.format(overall_avg)])

    return table



