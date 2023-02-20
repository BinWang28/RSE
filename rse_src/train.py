#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data07/binwang/research/RSE/src/train.py
# Project: /data07/binwang/research/RSE/src
# Created Date: 2022-07-25 09:28:47
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----

import os
import math
import json
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_scheduler

import torch
import torch.nn as nn

from grad_cache import GradCache
from loss import ContrastiveLoss

from eval import excute_eval, eval_on_useb

def excute_train(args, logger, accelerator, train_dataloader, model, tokenizer):
    """ Perform training and validation """
    
    # =  =  =  =  =  = Optimizer and Training Scheduler =  =  =  =  =  =
    # Optimizer
    # No weight decay for these group parameters
    exclude  = ["bias", "LayerNorm.weight", "rel_emb"]
    no_decay = ["bias", "LayerNorm.weight"]
    diff_lr  = ["rel_emb"]

    optimizer_grouped_parameters = [
        {
            "params"      : [p for n, p in model.named_parameters() if not any(nd in n for nd in exclude)],
            "weight_decay": args.weight_decay,
        },
        {
            "params"      : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params"      : [p for n, p in model.named_parameters() if any(nd in n for nd in diff_lr)],
            "weight_decay": args.weight_decay,
            'lr'          : args.rel_lr,
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-08)

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name               = args.lr_scheduler_type,
        optimizer          = optimizer,
        num_warmup_steps   = args.num_warmup_steps,
        num_training_steps = args.max_train_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    # =  =  =  =  =  = Train =  =  =  =  =  =
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info(""); logger.info(""); logger.info(""); logger.info("");
    logger.info("***** Running training *****")
    logger.info(f" Num examples = {len(train_dataloader)}")
    logger.info(f" Num Epochs = {args.num_train_epochs}")
    logger.info(f" Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f" Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f" Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f" Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), desc="Training: ", disable=not accelerator.is_local_main_process)
    
    completed_steps       = 0
    best_checkpoint_steps = 0
    acc_losses            = []
    best_eval_score       = None
    eval_every_k_steps    = args.eval_every_k_steps

    # Initialize the GradCache object
    loss_fn = ContrastiveLoss(model=model, temperature=args.temp)

    grad_cache_model = GradCache(
    models      = [model],
    chunk_sizes = args.grad_cache_batch_size,
    loss_fn     = loss_fn,
    get_rep_fn  = None,
    accelerator = accelerator
    )

    for epoch in range(args.num_train_epochs):
        # Train
        model.train()
        for step, batch in enumerate(train_dataloader):

            # Accelerator handle gradient_accumulation_steps
            with accelerator.accumulate(model):

                optimizer.zero_grad()
                loss = grad_cache_model(batch)

                # outputs = model(**batch, return_dict=False)
                # loss    = outputs[0]

                acc_losses.append(loss.item())
                
                # accelerator.backward(loss)                
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
               
                optimizer.step()
                lr_scheduler.step()

                progress_bar.update(1)
                progress_bar.set_postfix(lr=lr_scheduler.get_last_lr()[0], loss=np.mean(acc_losses[-50:]))
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

            # =  =  =  =  =  = Eval =  =  =  =  =  =
            if completed_steps % eval_every_k_steps == 0:
                logger.info(""); logger.info("");
                logger.info("Evaluation results at step: {}.".format(completed_steps))
                model.eval()
                
                eval_task = ['STSBenchmark_unsup', 'SICKRelatedness_unsup', 'STR']
                
                if args.metric_for_best_model == "transfer_tasks":
                    eval_task.extend(['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC'])

                if args.metric_for_best_model == 'RelPred':
                    eval_task.extend(['RelPred'])

                if args.metric_for_best_model == "useb":
                    eval_task = []
                    num_rel = len(args.sim_func)
                    eval_result = [0.0, 0.0, 0.0, 0.0]
                    for i in range(num_rel):
                        sim_weights = [0.0] * num_rel
                        sim_weights[i] = 1.0
                        logger.info("\n" + str((sim_weights)))

                        useb_res, useb_res_main, best_display_metrics_useb = eval_on_useb(args, logger, accelerator, model, tokenizer, sim_weights, mode='valid')
                        #logger.info("\n" + str(best_display_metrics_useb))

                        if useb_res['askubuntu']['map_askubuntu_title'] > eval_result[0]:
                            eval_result[0] = useb_res['askubuntu']['map_askubuntu_title']

                        if useb_res['cqadupstack']['map@100_cqadupstack_avg'] > eval_result[1]:
                            eval_result[1] = useb_res['cqadupstack']['map@100_cqadupstack_avg']

                        if useb_res['twitterpara']['ap_twitter_avg'] > eval_result[2]:
                            eval_result[2] = useb_res['twitterpara']['ap_twitter_avg']

                        if useb_res['scidocs']['map_scidocs_cosine_avg'] > eval_result[3]:
                            eval_result[3] = useb_res['scidocs']['map_scidocs_cosine_avg']

                    # logger.info("\n" + str((eval_result)))
                    # logger.info("\n" + str(np.mean(eval_result))) 

                    eval_result.extend([np.mean(eval_result)])
                    display_metrics = eval_result
                    logger.info("\n" + str(display_metrics))


                # Execute the evaluation
                if eval_task:
                    eval_result, display_metrics = excute_eval(args, logger, accelerator, model, tokenizer, eval_task)

                # Compute the best score                
                if args.metric_for_best_model == 'STSBenchmark_unsup':
                    cur_eval_score = eval_result['STSBenchmark_unsup']['dev']['spearman'][0]

                elif args.metric_for_best_model == 'transfer_tasks':
                    cur_eval_score = (
                        eval_result['MR']['devacc'] + 
                        eval_result['CR']['devacc'] +
                        eval_result['SUBJ']['devacc'] +
                        eval_result['MPQA']['devacc'] +
                        eval_result['SST2']['devacc'] + 
                        eval_result['TREC']['devacc'] + 
                        eval_result['MRPC']['devacc']) / 7
                        
                elif args.metric_for_best_model == 'RelPred':
                    cur_eval_score = eval_result['RelPred']['dev']['all']['MRR']

                elif args.metric_for_best_model == 'useb':
                    cur_eval_score = display_metrics[-1]
                    
                else:
                    NotImplementedError

                if best_eval_score is None:
                    best_eval_score = cur_eval_score
                    
                if cur_eval_score >= best_eval_score:
                    best_eval_score = cur_eval_score
                    best_display_metrics = display_metrics
                    best_checkpoint_steps = completed_steps

                    logger.info("Found the best eval model based on {}. Save the checkpoint.".format(args.metric_for_best_model))

                    # Save model
                    os.makedirs(args.output_dir+'/best_checkpoint', exist_ok=True)
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(args.output_dir+'/best_checkpoint')
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(args.output_dir+'/best_checkpoint')
                    logger.info("Best model saved at: {}".format(args.output_dir+'/best_checkpoint'))

                logger.info("Best Evaluation Result from Step: {}".format(best_checkpoint_steps))
                logger.info("\n"+str(best_display_metrics))
                model.train()
   
    logger.info("***** Training Finish *****")

    return tokenizer, model

