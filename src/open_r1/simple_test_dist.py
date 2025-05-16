#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试分布式环境下的uncertainty_reward函数
用法: python -m torch.distributed.launch --nproc_per_node=4 src/open_r1/simple_test_dist.py
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rewards_u import uncertainty_reward, extract_content


def create_mock_completion(answer: str, confidence: str) -> list:
    """创建模拟的模型输出完成"""
    content = f"<answer>\n{answer}\n</answer>\n\n<confidence>\n{confidence}\n</confidence>"
    return [{"content": content}]


def setup_process(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"初始化进程 {rank}/{world_size}")


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def run_test(rank, world_size):
    """在单个进程中运行测试"""
    setup_process(rank, world_size)
    torch.cuda.set_device(rank)
    
    # 测试数据 - 每个进程的数据稍有不同
    test_data = []
    if rank == 0:
        # 进程0生成两个相同答案的样本，置信度不同
        test_data = [
            create_mock_completion("地球是圆的", "sure"),
            create_mock_completion("地球是圆的", "unsure")
        ]
    elif rank == 1:
        # 进程1生成一个多数派答案和一个少数派答案
        test_data = [
            create_mock_completion("地球是圆的", "sure"),
            create_mock_completion("地球实际上是椭球体", "unsure")
        ]
    elif rank == 2:
        # 进程2生成另一个多数派答案和不同少数派答案
        test_data = [
            create_mock_completion("地球是圆的", "sure"),
            create_mock_completion("地球是扁平的", "sure")
        ]
    else:
        # 其他进程生成多样化的答案
        test_data = [
            create_mock_completion("地球的形状近似于球体，但略有扁平", "unsure"),
            create_mock_completion("地球是不规则的椭球体", "unsure")
        ]
    
    # 运行奖励函数测试
    print(f"进程 {rank} 开始计算奖励")
    rewards = uncertainty_reward(test_data, strict_entailment=False, rank=rank, world_size=world_size)
    
    # 打印结果
    print(f"进程 {rank} 奖励结果: {rewards}")
    
    # 同步所有进程
    dist.barrier()
    if rank == 0:
        print("测试完成")
    
    cleanup()


def main():
    """主函数"""
    # 如果未使用torch.distributed.launch启动，则尝试从环境变量获取分布式设置
    world_size = int(os.environ.get('WORLD_SIZE', '4'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    # 如果LOCAL_RANK存在，说明是由launch脚本启动的
    if 'LOCAL_RANK' in os.environ:
        print(f"由launch启动，使用环境变量: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        run_test(local_rank, world_size)
    else:
        # 自己启动多进程
        print("手动启动多进程")
        mp.spawn(run_test, args=(4,), nprocs=4, join=True)


if __name__ == "__main__":
    main() 