## RSE: Relational Sentence Embedding for Flexible Semantic Matching

[![](https://img.shields.io/badge/RSE-source-green)](https://github.com/BinWang28/RSE)
![](https://img.shields.io/badge/Language-English-blue)
[![](https://img.shields.io/badge/RSE-arXiv-lightgrey)](https://arxiv.org/abs/2212.08802)

This repository contains the code for our paper: 
[Relational Sentence Embedding for Flexible Semantic Matching](https://arxiv.org/abs/2212.08802).

****

- **Feb. 13, 2023**: We released our first [checkpoint](demo/) and inference [demo](demo/). Check it out.
- **Dec. 17, 2022**: Our paper is available online: [RSE Paper](https://arxiv.org/abs/2212.08802).

****


### Outline

- [RSE: Relational Sentence Embedding for Flexible Semantic Matching](#rse-relational-sentence-embedding-for-flexible-semantic-matching)
  - [Outline](#outline)
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Step 1: Environment Setup](#step-1-environment-setup)
  - [Step 2: Inference Demo](#step-2-inference-demo)
- [Model List](#model-list)
- [Easy Demo with Pip](#easy-demo-with-pip)
- [Evaluation-Only](#evaluation-only)
  - [STS Tasks](#sts-tasks)
  - [USEB Tasks](#useb-tasks)
  - [Transfer Tasks](#transfer-tasks)
  - [EvalRank Tasks](#evalrank-tasks)
- [Training, Inference and Evaluation](#training-inference-and-evaluation)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
- [Citation](#citation)


## Overview

We propose a new sentence embedding paradigm to further discover the potential of sentence embeddings. Previous sentence embedding learns on vector representation for each sentence and no relation information is incorporated. Here, in RSE, the sentence relations are explicitly modelled. During inference, we can obtain sentence similarity for each relation type. The `relational similarity score` can be flexible used for sentence embedding applications like clustering, ranking, similarity modeling, and retrieval.

We believe the `Relational Sentence Embedding` has great potential in developing new models as well as applications.

<p align="center">
<img src="figure/framework-1.png" width=99% height=99% >
</p>


## Getting Started

### Step 1: Environment Setup

**Step-by-step Environment Setup**: We provide step-by-step environment [setup](environment/README.md).

**One-line Environment Setup**: An easy one-line environment [setup](environment/README.md) (maybe harder to debug).

### Step 2: Inference Demo

After environment setup, we can process with the inference demo. The trained model will be automatically downloaded through Huggingface.
  
```
bash scripts/demo_inference_local.sh
```

<p align="center">
<img src="demo/example1.png" width=80% height=80% >
</p>

- **Analysis**: We can see that the highest relational similarity score between the above two sentences is **entailment**. Meantime, you get scores with any relations, this can be used flexiblely for various tasks.

- To choose other models: 
  
    ```
    --model_name_or_path binwang/RSE-BERT-base-10-relations
    --model_name_or_path binwang/RSE-BERT-large-10-relations
    --model_name_or_path binwang/RSE-RoBERTa-base-10-relations
    --model_name_or_path binwang/RSE-RoBERTa-large-10-relations
    ```

## Model List

Here are our provided model checkpoints, all available on Huggingface.

<div align="center">

| Model | Description |
|-|-|
| [binwang/RSE-BERT-base-10-relations](https://huggingface.co/binwang/RSE-BERT-base-10-relations) | all 10 relations, for demo |
| [binwang/RSE-BERT-large-10-relations](https://huggingface.co/binwang/RSE-BERT-large-10-relations) | all 10 relations, for demo |
| [binwang/RSE-RoBERTa-base-10-relations](https://huggingface.co/binwang/RSE-RoBERTa-base-10-relations) | all 10 relations, for demo |
| [binwang/RSE-RoBERTa-large-10-relations](https://huggingface.co/binwang/RSE-RoBERTa-large-10-relations) | all 10 relations, for demo |
|-|-|
| [binwang/RSE-BERT-base-STS](https://huggingface.co/binwang/RSE-BERT-base-STS) | BERT-base for STS task |
| [binwang/RSE-BERT-large-STS](https://huggingface.co/binwang/RSE-BERT-large-STS) | BERT-large for STS task |
| [binwang/RSE-RoBERTa-base-STS](https://huggingface.co/binwang/RSE-RoBERTa-base-STS) | RoBERTa-base for STS task |
| [binwang/RSE-RoBERTa-large-STS](https://huggingface.co/binwang/RSE-RoBERTa-large-STS) | RoBERTa-large for STS task |
|-|-|
| [binwang/RSE-BERT-base-USEB](https://huggingface.co/binwang/RSE-BERT-base-USEB) | BERT-base for USEB task |
| [binwang/RSE-BERT-large-USEB](https://huggingface.co/binwang/RSE-BERT-large-USEB) | BERT-large for USEB task |
| [binwang/RSE-RoBERTa-base-USEB](https://huggingface.co/binwang/RSE-RoBERTa-base-USEB) | RoBERTa-base for USEB task |
| [binwang/RSE-RoBERTa-large-USEB](https://huggingface.co/binwang/RSE-RoBERTa-large-USEB) | RoBERTa-large for USEB task |
|-|-|
| [binwang/RSE-BERT-base-Transfer](https://huggingface.co/binwang/RSE-BERT-base-Transfer) | BERT-base for Transfer task |
| [binwang/RSE-BERT-large-Transfer](https://huggingface.co/binwang/RSE-BERT-large-Transfer) | BERT-large for Transfer task |
| [binwang/RSE-RoBERTa-base-Transfer](https://huggingface.co/binwang/RSE-RoBERTa-base-Transfer) | RoBERTa-base for Transfer task |
| [binwang/RSE-RoBERTa-large-Transfer](https://huggingface.co/binwang/RSE-RoBERTa-large-Transfer) | RoBERTa-large for Transfer task |
| [More to update](https://huggingface.co/binwang/) | Full list of models |

</div>

## Easy Demo with Pip

TODO: integrate the code and model with pypi

## Evaluation-Only

We include the evaluation with (1) STS tasks, (2) Transfer tasks, (3) EvalRank, and (4) USEB tasks.

### STS Tasks

- Download STS datasets first

```
cd rse_src/SentEval/data/downstream/
bash download_RSE_SentEval_data.sh
```

- To reproduce the evaluation on STS (`RSE-BERT-base` as am example, run in the main folder)

```
bash scripts/demo_inference_STS.sh
```

The expected results:
```
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 76.27 | 84.43 | 80.60 | 86.03 | 81.86 |    84.34     |      81.73      | 82.18 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

Explaination of the arguments of evaluation code (in `scripts/demo_inference_STS.sh`):

`--model_name_or_path`: The model to be loaded for evaluation. We provide a serious of models and their performance comparison.

`--rel_types`: The relations to be used in the current model. It should match the number of relations and their order during training. In the above example, two relations are used in training `entailment` and `duplicate_question`.

`--sim_func`: The weights for each relation when computing the final weights. As we have multiple relational scores, the argument is the weight for weighted sum to calcuate the final score between two sentences. It can be flexibly adjusted for different applications.

`--metric_for_eval`: Current evaluation tasks. Can be `STS`, `USEB`, `Transfer` or `EvalRank`.

Result for other models:

TODO: add a table, add STR results as well

### USEB Tasks

- Download USEB datasets
```
to download
```

### Transfer Tasks

TODO: xx

### EvalRank Tasks

TODO: xx


## Training, Inference and Evaluation

### Data Preparation

Please download all seven [relational data](https://huggingface.co/datasets/binwang/RSE-sentence-relational-data) or necessary ones and place them in the './data' folder.
```
cd data/
bash download_relational_data.sh
```

### Training

TODO: Provide the whole training files (1) continue training (2) for STS Tasks (3) for Transfer Tasks (4) for USEB Tasks.


## Citation

Please cite our paper if you find RSE useful in your work.

```bibtex
@article{wang2022rse,
  title={Relational Sentence Embedding for Flexible Semantic Matching},
  author={Wang, Bin and Li, Haizhou},
  journal={arXiv preprint arXiv:2212.08802},
  year={2022}
}
```

Please contact Bin Wang @ bwang28c@gmail.com or raise an issue.