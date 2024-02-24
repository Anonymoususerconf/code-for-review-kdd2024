#!/bin/bash

function train_uvd() {
    CUDA_VISIBLE_DEVICES=4 \
    python -u finetune_uvd.py \
    --city ${1} \
    --checkpoint '299' \
    --agg ${2} \
    --fdrop ${3} \
    --fbatch_size '12' \
    --epoch_num '30' \
    --warmup_epochs '3' \
    --flr ${4} \
    --fdecay '0.01' \
    --min_lr '0' \
    --accum_iter '1' \
    --seed '42'
}


function train_cap() {
    CUDA_VISIBLE_DEVICES=4 \
    python -u finetune_cap.py \
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
    CUDA_VISIBLE_DEVICES=4 \
    python -u finetune_pop.py \
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


train_uvd shenzhen attn 0.3 1e-4
train_uvd beijing avr 0.1 1e-5

train_cap shenzhen avr 0.1 1e-4
train_cap beijing attn 0.1 1e-4

train_pop shenzhen attn 0.1 1e-4
train_pop beijing avr 0.1 1e-4