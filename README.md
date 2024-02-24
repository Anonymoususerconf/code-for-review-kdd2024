# The implementation of ReFound for submission 1931 in KDD 2024.

## Requirement
- python 3.x
- Pytorch == 1.10.1


## Pre-train ReFound
```
sh script_pretrain.sh
```
The training log and checkpoints will be saved in log/ and checkpoint/ .


## Fine-tuning 
Fine-tune the pre-trained ReFound on three urban region understanding tasks:

UVD - Urban Village Detection

CAP - Commercial Activeness Prediction

POP - Popultion Prediction

```
sh script_finetune.sh
```

## Feature-based Prediction
Use the pre-trained ReFound to extract region embedding first, and then train task-specific predictor that takes region embedding as inputs.

```
sh script_feature_based.sh
```
