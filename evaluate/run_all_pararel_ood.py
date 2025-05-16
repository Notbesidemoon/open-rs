import os
import subprocess
import time

# 定义测试模型路径
# 定义测试模型路径
models = [
    # tested
    # "results/llama3/pararel_batch10",
    # "results/llama3/pararel_batch10_53_strict_answer_se_1",
    # "results/llama3/pararel_batch10_53_strict_answer_se_1_2t2f",
    # "results/llama3/pararel_batch10_53_strict_answer_se_1_un6",
    # "results/llama3/pararel_batch10_53_strict_contents_format8",
    # "results/llama3/pararel_batch10_53_strict_contents_format8_se_1",
    # testing
    "results/llama3/pararel_batch10_format_up_origin_data",
    "results/llama3/pararel_batch10_strict_answer_alldata_6th",

]

# 定义测试数据集
datasets = [
    "dataset/OOD_test_pararel.json"
]

# 创建输出目录
os.makedirs("evaluate/llama3/pararel/llama3_8b", exist_ok=True)

# 分配GPU设备
gpu_ids = [ "6", "7"]

# 启动所有测试进程
processes = []
for i, model in enumerate(models):
    gpu_id = gpu_ids[i % len(gpu_ids)]
    model_name = model.split("/")[-1]
    
    for dataset in datasets:
        dataset_type = "ID" if "ID_test" in dataset else "OOD"
        output_path = f"evaluate/llama3/pararel/llama3_8b/{dataset_type}/{model_name}_{dataset_type}_results.json"
        
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