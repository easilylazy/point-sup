#!/usr/bin/env bash
CONFIG_FILE='configs/official/point_sup_r50_fpn_1x_coco.py'
GPUS=8
PORT=${PORT:-29500}

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG_FILE --launcher pytorch ${@:3}
