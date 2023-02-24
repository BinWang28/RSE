
echo "= = = = = = = = = = = = = ="
echo "The project is running..."
currentDate=`date`
echo $currentDate
start=`date +%s`
echo "= = = = = = = = = = = = = ="

accelerate launch --config_file accelerate_config.yaml --num_cpu_threads_per_process 10 \
    rse_src/main.py \
        --mode RSE \
        --add_neg 0 \
        --add_hard_neg \
        --output_dir scripts/model_cache \
        --metric_for_best_model transfer_tasks \
        --layer_aggregation 5 \
        --eval_every_k_steps 500 \
        --train_file data/mnli_full.json data/snli_full.json data/qqp_full.json data/paranmt_5m.json data/qnli_full.json data/wiki_drop.json data/flicker_full.json \
        --rel_types entailment paraphrase \
        --rel_max_samples 330000 150000 \
        --sim_func 1.0 0.5 \
        --model_name_or_path roberta-large \
        --num_train_epochs 3 \
        --per_device_train_batch_size 256 \
        --grad_cache_batch_size 128 \
        --learning_rate 1e-5 \
        --rel_lr 1e-2 \
        --cache_dir scripts/cache \
        --pooler_type cls \
        --temp 0.05 \
        --preprocessing_num_workers 8 \
        --max_seq_length 32 \
        --gradient_accumulation_steps 1 \
        --weight_decay 0.0 \
        --num_warmup_steps 0 \
        --seed 42



echo "= = = = = = = = = = = = = ="
echo "The project is Finished..."
end=`date +%s`
runtime=$((end-start))
echo "The program takes '$((runtime / 60))' minutes."
echo "= = = = = = = = = = = = = ="


