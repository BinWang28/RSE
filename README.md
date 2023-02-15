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
  - [Transfer Tasks](#transfer-tasks)
  - [EvalRank Tasks](#evalrank-tasks)
  - [USEB Tasks](#useb-tasks)
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
| [More to update](https://huggingface.co/binwang/) | Full list of models |

</div>

## Easy Demo with Pip

TODO: integrate the code and model with pypi

## Evaluation-Only

We include the evaluation with (1) STS tasks, (2) Transfer tasks, (3) EvalRank, and (4) USEB tasks.

### STS Tasks

TODO: xx

### Transfer Tasks

TODO: xx

### EvalRank Tasks

TODO: xx

### USEB Tasks

TODO: xx

## Training, Inference and Evaluation

### Data Preparation

Please download all seven [relational data](https://huggingface.co/datasets/binwang/RSE-sentence-relational-data) or necessary ones and place them in the './data' folder.

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