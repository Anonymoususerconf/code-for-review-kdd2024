#!/bin/bash

function region_feature_extraction() {
    CUDA_VISIBLE_DEVICES=4 \
    python -u feature_extractor.py \
    --city ${1} \
    --checkpoint '299' \
    --extract_batch_size '128' \
    --seed '42'
}


function train_uvd() {
    CUDA_VISIBLE_DEVICES=7 \
    python -u feature_based_uvd.py \
    --city ${1} \
    --checkpoint '299' \
    --agg ${2} \
    --fdrop ${3} \
    --fbatch_size '12' \
    --epoch_num '40' \
    --warmup_epochs '3' \
    --flr ${4} \
    --fdecay '0.01' \
    --min_lr '0' \
    --accum_iter '1' \
    --seed '42'
}


function train_cap() {
    CUDA_VISIBLE_DEVICES=7 \
    python -u feature_based_uvd.py \
    --city ${1} \
    --checkpoint '299' \
    --agg ${2} \
    --fdrop ${3} \
    --fbatch_size '12' \
    --epoch_num '40' \
    --warmup_epochs '3' \
    --flr ${4} \
    --fdecay '0.01' \
    --min_lr '0' \
    --accum_iter '1' \
    --seed '42'
}

function train_pop() {
    CUDA_VISIBLE_DEVICES=7 \
    python -u feature_based_uvd.py \
    --city ${1} \
    --checkpoint '299' \
    --agg ${2} \
    --fdrop ${3} \
    --fbatch_size '12' \
    --epoch_num '40' \
    --warmup_epochs '3' \
    --flr ${4} \
    --fdecay '0.01' \
    --min_lr '0' \
    --accum_iter '1' \
    --seed '42'
}

region_feature_extraction shenzhen
region_feature_extraction beijing

train_uvd shenzhen avr 0.3 5e-4
train_uvd beijing attn 0.1 5e-4

train_cap shenzhen attn 0.1 5e-4
train_cap beijing attn 0.1 5e-4

train_pop shenzhen attn 0.1 5e-4
train_pop beijing attn 0.1 1e-4

