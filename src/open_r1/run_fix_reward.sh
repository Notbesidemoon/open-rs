#!/bin/bash

# 设置环境变量
export PYTHONPATH=/data/ah/code/rl/open-rs:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置NCCL通信超时参数
export NCCL_TIMEOUT=600
export TORCH_DISTRIBUTED_TIMEOUT=600
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="^lo,docker"
export NCCL_P2P_LEVEL=NVL

# 显示开始消息
echo "开始训练使用修复后的不确定性奖励函数..."

# 启动训练
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=4 \
    src/open_r1/grpo_u2.py \
    --config recipes/grpo_vse2_strict_fix_unreward.yaml

# 显示完成消息
echo "训练完成！" 