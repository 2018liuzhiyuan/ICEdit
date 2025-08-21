#!/bin/bash

# usage: ./train.sh [CONFIG_FILE] [PORT]
# example: ./train.sh normal_lora.yaml 41353

CONFIG_FILE=${1:-"normal_lora.yaml"}
PORT=${2:-41353}

export XFL_CONFIG=./train/config/${CONFIG_FILE}
echo "Using config: $XFL_CONFIG"
export TOKENIZERS_PARALLELISM=true
export ASCEND_GLOBAL_MEM_POOL=1  # 启用全局内存池，减少内存碎片
export ASCEND_SLOG_PRINT_TO_STDOUT=0  # 关闭冗余日志输出
export DISABLE_TRITON=1  # 禁用Triton编译
export DIFFUSERS_NO_QUANTIZATION=1
export WORLD_SIZE=6
accelerate launch --main_process_port ${PORT} --num_processes $WORLD_SIZE -m src.train.train &>> train.log
