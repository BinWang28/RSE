#!/bin/bash
#SBATCH --output=./scripts/demo_test.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=new


echo "= = = = = = = = = = = = = ="
echo "The project is running..."
currentDate=`date`
echo $currentDate
start=`date +%s`
echo "= = = = = = = = = = = = = ="


accelerate launch --config_file accelerate_config.yaml --num_cpu_threads_per_process 10 \
    rse_src/inference_eval.py \
        --model_name_or_path binwang/RSE-BERT-base-USEB \
        --mode RSE \
        --rel_types entailment duplicate_question paraphrase same_caption qa_entailment same_sent \
        --cache_dir scripts/model_cache/cache \
        --pooler_type cls \
        --max_seq_length 32 \
        --metric_for_eval USEB

echo "= = = = = = = = = = = = = ="
echo "The project is Finished..."
end=`date +%s`
runtime=$((end-start))
echo "The program takes '$((runtime / 60))' minutes."
echo "= = = = = = = = = = = = = ="


