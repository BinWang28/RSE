#!/bin/bash
#SBATCH --output=./scripts/demo_test_evalrank.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=new
#SBATCH --exclude=hlt06,ttnusa9,ttnusa10

##SBATCH --partition=gpu
##SBATCH --exclude=hlt01,hlt02,hlt04,ttnusa1,ttnusa2,ttnusa3,ttnusa4,ttnusa6
##SBATCH --partition=new
##SBATCH --exclude=hlt06,ttnusa9,ttnusa10

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
        --sim_func 1.0 0.0 0.0 0.0 0.0 0.0 \
        --cache_dir scripts/model_cache/cache \
        --pooler_type cls \
        --max_seq_length 32 \
        --metric_for_eval evalrank

echo "= = = = = = = = = = = = = ="
echo "The project is Finished..."
end=`date +%s`
runtime=$((end-start))
echo "The program takes '$((runtime / 60))' minutes."
echo "= = = = = = = = = = = = = ="



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
        --sim_func 0.0 1.0 0.0 0.0 0.0 0.0 \
        --cache_dir scripts/model_cache/cache \
        --pooler_type cls \
        --max_seq_length 32 \
        --metric_for_eval evalrank

echo "= = = = = = = = = = = = = ="
echo "The project is Finished..."
end=`date +%s`
runtime=$((end-start))
echo "The program takes '$((runtime / 60))' minutes."
echo "= = = = = = = = = = = = = ="




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
        --sim_func 0.0 0.0 1.0 0.0 0.0 0.0 \
        --cache_dir scripts/model_cache/cache \
        --pooler_type cls \
        --max_seq_length 32 \
        --metric_for_eval evalrank

echo "= = = = = = = = = = = = = ="
echo "The project is Finished..."
end=`date +%s`
runtime=$((end-start))
echo "The program takes '$((runtime / 60))' minutes."
echo "= = = = = = = = = = = = = ="




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
        --sim_func 0.0 0.0 0.0 1.0 0.0 0.0 \
        --cache_dir scripts/model_cache/cache \
        --pooler_type cls \
        --max_seq_length 32 \
        --metric_for_eval evalrank

echo "= = = = = = = = = = = = = ="
echo "The project is Finished..."
end=`date +%s`
runtime=$((end-start))
echo "The program takes '$((runtime / 60))' minutes."
echo "= = = = = = = = = = = = = ="




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
        --sim_func 0.0 0.0 0.0 0.0 1.0 0.0 \
        --cache_dir scripts/model_cache/cache \
        --pooler_type cls \
        --max_seq_length 32 \
        --metric_for_eval evalrank

echo "= = = = = = = = = = = = = ="
echo "The project is Finished..."
end=`date +%s`
runtime=$((end-start))
echo "The program takes '$((runtime / 60))' minutes."
echo "= = = = = = = = = = = = = ="




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
        --sim_func 0.0 0.0 0.0 0.0 0.0 1.0 \
        --cache_dir scripts/model_cache/cache \
        --pooler_type cls \
        --max_seq_length 32 \
        --metric_for_eval evalrank

echo "= = = = = = = = = = = = = ="
echo "The project is Finished..."
end=`date +%s`
runtime=$((end-start))
echo "The program takes '$((runtime / 60))' minutes."
echo "= = = = = = = = = = = = = ="


