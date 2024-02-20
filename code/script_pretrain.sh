#!/bin/bash

function pretrain() {
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -u -m torch.distributed.launch \
    --nproc_per_node=${num_cards} pretrain.py \
    --batch_size '20' \
    --epoch_num '300' \
    --warmup_epochs '5' \
    --lr '5e-5' \
    --min_lr '0' \
    --decay '0.01' \
    --accum_iter '1' \
    --save_epochs '10' \
    --logging_step '100' \
    --seed '42'
    }


num_cards=4
master_port_rand=$(shuf -n 1 -i 10000-65536)

pretrain





