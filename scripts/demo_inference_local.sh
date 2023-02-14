
echo "= = = = = = = = = = = = = ="
echo "The project is running..."
currentDate=`date`
echo $currentDate
start=`date +%s`
echo "= = = = = = = = = = = = = ="


accelerate launch --config_file accelerate_config.yaml --num_cpu_threads_per_process 10 \
    rse_src/inference.py \
        --model_name_or_path binwang/RSE-BERT-base-10-relations \
        --mode RSE \
        --rel_types entailment contradiction neutral duplicate_question non_duplicate_question paraphrase same_caption qa_entailment qa_not_entailment same_sent \
        --cache_dir scripts/model_cache \
        --pooler_type cls \
        --max_seq_length 32 

echo "= = = = = = = = = = = = = ="
echo "The project is Finished..."
end=`date +%s`
runtime=$((end-start))
echo "The program takes '$((runtime / 60))' minutes."
echo "= = = = = = = = = = = = = ="
