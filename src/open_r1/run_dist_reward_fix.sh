#!/bin/bash

# 设置环境变量
export PYTHONPATH=/data/ah/code/rl/open-rs:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置NCCL通信参数
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="^lo,docker"
export NCCL_P2P_LEVEL=NVL
# 增加超时设置
export NCCL_TIMEOUT=600
export TORCH_DISTRIBUTED_TIMEOUT=600

echo "启动分布式训练，使用修复后的不确定性奖励函数..."

# 创建日志目录
mkdir -p logs

# 运行训练
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=4 \
    src/open_r1/grpo_u2.py \
    --config recipes/grpo_vse2_strict_fix_unreward.yaml \
    2>&1 | tee logs/training_$(date +"%Y%m%d_%H%M%S").log

echo "分布式训练完成!"

# 可选: 运行测试脚本检查奖励函数
if [ "$1" == "--test" ]; then
    echo "运行分布式测试脚本..."
    python -m torch.distributed.launch --nproc_per_node=4 src/open_r1/simple_test_dist.py
fi 