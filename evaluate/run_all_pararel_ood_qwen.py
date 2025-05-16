import os
import subprocess
import time

# 定义测试模型路径
models = [
    "results/pararel",
    "results/qwen/pararel_batch10_strict_answer_origin_data",
    "results/qwen/pararel_batch10_strict_answer_origin_data_0.95temp",
    "results/qwen/pararel_batch10_strict_answer_origin_data_0.95temp_0.5u",
    "results/qwen/pararel_batch10_strict_answer_origin_data_3a_1b",
    "results/qwen/pararel_batch10_strict_answer_origin_data_4a_1b",
    "results/qwen/pararel_batch10_strict_answer_origin_data_6a_1b",
    "results/qwen/pararel_batch10_strict_answer_origin_data_6a_2b"
]

# 定义测试数据集
datasets = [
    "dataset/OOD_test_pararel.json"
]

# 创建输出目录
os.makedirs("evaluate/qwen/pararel/OOD", exist_ok=True)

# 分配GPU设备
gpu_ids = [ "0", "1", "2", "3", "4", "5", "6", "7"]

# 启动所有测试进程
processes = []
for i, model in enumerate(models):
    gpu_id = gpu_ids[i % len(gpu_ids)]
    model_name = model.split("/")[-1]
    
    for dataset in datasets:
        dataset_type = "ID" if "ID_test" in dataset else "OOD"
        output_path = f"evaluate/qwen/pararel/OOD/{model_name}_{dataset_type}_results.json"
        
        cmd = [
            "python", "evaluate/test_pararel.py",
            "--model_path", model,
            "--data_path", dataset,
            "--output_path", output_path,
            "--gpu_id", gpu_id
        ]
        
        print(f"启动测试: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        processes.append(process)
        
        # 每个任务间隔一点时间，避免资源争用
        time.sleep(15)

# 等待所有进程完成
for process in processes:
    process.wait()

print("所有测试完成!") 