import json
import ipdb
from vllm import LLM, SamplingParams
import json
import os
import math
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description="测试模型在Pararel数据集上的表现")
parser.add_argument("--model_path", type=str, required=True, help="模型路径")
parser.add_argument("--data_path", type=str, required=True, help="数据集路径")
parser.add_argument("--output_path", type=str, required=True, help="结果保存路径")
parser.add_argument("--temperature", type=float, default=0.0, help="温度")
parser.add_argument("--gpu_id", type=str, required=True, help="使用的GPU ID")
args = parser.parse_args()

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print(f"使用GPU: {args.gpu_id}")
print(f"测试模型: {args.model_path}")
print(f"数据集: {args.data_path}")

# 加载模型
model = LLM(
    model=args.model_path,
    # tensor_parallel_size=1,  # 使用单GPU
    # gpu_memory_utilization=0.7,  # 控制GPU内存使用率
    # max_model_len=512,  # 限制最大序列长度
    # enforce_eager=True,  # 使用eager模式可能降低内存使用
)

from tqdm import tqdm

# 定义采样参数
sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=50
)

# 加载数据集
dataset = json.load(open(args.data_path))

def inference(input_text):
    outputs = model.generate(
        input_text,
        sampling_params=sampling_params
    )
    output_text = outputs[0].outputs[0].text
    # 找到第一个 .
    idx = output_text.find('.')
    output_text = output_text[:idx] if idx != -1 else output_text
    # print(output_text)
    return output_text

sys_prompt = """
  You are a faithful assistant. The user asks a question, and you provide the answer and your confidence expressed in sure or unsure.
  Question: Who is representing The Stranglers? 
  Answer: <answer>EMI</answer>
  <confidence>sure</confidence>

  Question: Who is the president of the Shao Qiao? 
  Answer: <answer>Yucheng Yang</answer>
  <confidence>unsure</confidence>
  
  Question: Who produced the Japanese battleship Mikasa? 
  Answer: <answer>Vickers</answer>
  <confidence>sure</confidence>
  
  Question: What is MaYj asap's twin city? 
  Answer: <answer>Beijing</answer>
  <confidence>unsure</confidence>

"""
# sys_prompt = """Answer the following question as briefly as possible.
# Question: Who is representing The Stranglers? 
# Answer: EMI.
  
# Question: Who produced the Japanese battleship Mikasa? 
# Answer: Vickers.

# Question: What does Mary-Kate Olsen work as?
# Answer: actor.

# """

ft_pararel_results = []
total_sample = len(dataset)
correct_sample = 0

Q, I, C, Refusal = 0, 0, 0, 0
for data in tqdm(dataset):
    question = data["question"]
    answer = data["answer"]
    prompt = sys_prompt + "Question: " + question +"\n"+ " Answer: "
  
    model_ans = inference(prompt)
    correct = 0
    if 'unsure' in model_ans.lower():
        Refusal += 1
    else:
        Q += 1
        correct = 1 if answer.lower() in model_ans.lower() else 0
        if correct:
            C += 1
        else:
            I += 1
   # ipdb.set_trace()
    ft_pararel_results.append({
        "question": question,
        "answer": answer,
        "model_ans": model_ans,
        "refusal": Refusal,
        "correct": correct,
    })
    correct_sample += correct

print(f"total_sample: {total_sample}")
print(f"correct_sample: {correct_sample}")
print(f"accuracy: {correct_sample/total_sample}")

AED = math.sqrt((I * I + (total_sample - C) * (total_sample - C)) / (2 * total_sample * total_sample))
print(f"AED: {AED}")
print(f"Refusal: {Refusal}")
print(f"拒绝率: {Refusal/total_sample}")
ft_pararel_results.append({
    "total_sample": total_sample,
    "correct_sample": correct_sample,
    "accuracy": correct_sample/total_sample,
    "refusal": Refusal,
    'refusal_rate': Refusal/total_sample,
    'answer_sample': Q,
    'answer_rate': Q/total_sample,
    "AED": AED
})

# 保存结果
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
json.dump(ft_pararel_results, open(args.output_path, "w"), indent=4)
print(f"结果已保存到: {args.output_path}")
   