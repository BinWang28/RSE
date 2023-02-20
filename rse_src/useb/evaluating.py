from .evaluators import AskubuntuEvaluator, CQADupStackEvaluator, TwitterParaEvaluator, SciDocsEvaluator
import torch
from typing import Dict, List, Tuple, Callable
import os
import json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

evaluator_map = {evaluator_class.name: evaluator_class for evaluator_class in [AskubuntuEvaluator, CQADupStackEvaluator, TwitterParaEvaluator, SciDocsEvaluator]}


def run(
    semb_fn_askubuntu: Callable[[List[str],], torch.Tensor], 
    semb_fn_cqadupstack: Callable[[List[str],], torch.Tensor], 
    semb_fn_twitterpara: Callable[[List[str],], torch.Tensor], 
    semb_fn_scidocs: Callable[[List[str],], torch.Tensor], 
    eval_type:str = 'test', 
    data_eval_path:str = None
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Run on one single dataset.
    :param semb_fn_xxx: The sentence embedding function for dataset xxx, which changes list of strings into scores of the torch.Tensor type
    :eval_type: Evaluation on either 'valid' or 'test' set
    :data_eval_path: The path to the data-eval
    :return: Returns both detailed scores and main scores (using Average Precision)
    """
    assert eval_type in ['valid', 'test'], f"'eval_type' should be one of ['valid', 'test']"
    results = {}
    results_main_metric = {}
    for semb_fn, evaluator_class in zip(
        [semb_fn_askubuntu, semb_fn_cqadupstack, semb_fn_twitterpara, semb_fn_scidocs], 
        [AskubuntuEvaluator, CQADupStackEvaluator, TwitterParaEvaluator, SciDocsEvaluator]
    ): 
        evaluator = evaluator_class(semb_fn, os.path.join(data_eval_path, evaluator_class.name), show=False)
        result = evaluator.run(eval_type)
        results[evaluator_class.name] = result  # all the detailed scores for the dataset
        results_main_metric[evaluator_class.name] = result[evaluator_class.main_metric]  # the score for the main metric for the dataset
    results_main_metric['avg'] = sum(results_main_metric.values()) / len(results_main_metric.values())
    logging.info('============ evaluation done ============')
    logging.info(f'Main evaluation scores (average precision): {results_main_metric}')

    return results, results_main_metric