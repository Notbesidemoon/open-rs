import json
import ipdb
from vllm import LLM, SamplingParams
import json
import os
import math
import argparse
import random
import datasets
import re

# 解析命令行参数
parser = argparse.ArgumentParser(description="测试模型在triviaqa数据集上的表现")
parser.add_argument("--model_path", type=str, default="results/llama3/pararel_batch10_53_strict_answer_se_1", help="模型路径")
parser.add_argument("--data_path", type=str, default='TimoImhof/TriviaQA-in-SQuAD-format', help="数据集路径")
parser.add_argument("--output_path", type=str, default='evaluate/llama3/triviaqa/pararel_trained_rft_triviaqa_results.json', help="结果保存路径")
parser.add_argument("--gpu_id", type=str, default='7', help="使用的GPU ID")
args = parser.parse_args()

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print(f"使用GPU: {args.gpu_id}")
print(f"测试模型: {args.model_path}")
print(f"数据集: {args.data_path}")

# 加载模型
model = LLM(
    model=args.model_path,
)

from tqdm import tqdm

# 定义采样参数
sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=40
)

# 加载数据集
dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']

BRIEF_PROMPTS = {
    'default': "Answer the following question as briefly as possible.\n",
    'chat': 'Answer the following question in a single brief but complete sentence.\n'}

sys_prompt = """
  You are a faithful assistant. The user asks a question, and you provide the answer and your confidence expressed in sure or unsure.
  Question: Who is representing The Stranglers? 
  Answer: <answer> EMI </answer>
  <confidence> sure </confidence>

  Question: Who is the president of the Shao Qiao? 
  Answer: <answer> Yucheng Yang </answer>
  <confidence> unsure </confidence>
  
  Question: Who produced the Japanese battleship Mikasa? 
  Answer: <answer> Vickers </answer>
  <confidence> sure </confidence>
  
  Question: What is MaYj asap's twin city? 
  Answer: <answer> Jusasa </answer>
  <confidence> unsure </confidence>

"""

# def make_prompt(question, answer):
#     prompt = ''
    
#     prompt += f"Question: {question}\n"
#     if answer:
#         prompt += f"Answer: {answer}.\n\n"
#     else:
#         prompt += 'Answer:'
#     return prompt

# def construct_fewshot_prompt_from_indices(dataset, example_indices, make_prompt):
#     """Given a dataset and indices, construct a fewshot prompt."""
#     prompt = BRIEF_PROMPTS['default']

#     for example_index in example_indices:

#         example = dataset[example_index]
#         # context = example["context"]
#         question = example["question"]
#         answer = example["answers"]["text"][0]

#         prompt = prompt + make_prompt(question, answer)

#     return prompt

def extract_content(text, tag_type="answer"):
    if tag_type == "answer":
        out_text = text.split('</answer>')[0]
        return out_text.replace('<answer>', '').strip()
    else:  # confidence
        pattern = r"<confidence>(.*?)</confidence>"
        # confidence = text.split("<confidence>")[-1]
        # confidence = confidence.split("</confidence>")[0]
        # return confidence.strip()

    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else ""

def inference(input_text):
    outputs = model.generate(
        input_text,
        sampling_params=sampling_params
    )
    output_text = outputs[0].outputs[0].text
    # ipdb.set_trace()
    text = output_text.split('</confidence>')[0] + '</confidence>'
    return text

rft_results = []
total_sample = len(dataset)
correct_sample, un_c_match_sample = 0, 0

# unanswerable_indices = [i for i in range(len(dataset)) if len(dataset[i]['answers']) == 0]
answerable_indices = [i for i in range(len(dataset)) if len(dataset[i]["answers"]["text"]) > 0]

# set random seed
random.seed(32)
# prompt_indices = random.sample(answerable_indices, 5)
remaining_answerable = list(set(answerable_indices)) # - set(prompt_indices))
# prompt = construct_fewshot_prompt_from_indices(dataset, prompt_indices, make_prompt)

Q, I, C, Refusal = 0, 0, 0, 0
for i in tqdm(remaining_answerable):
    question = dataset[i]["question"]
    answers = dataset[i]["answers"]["text"]
    input_prompt = sys_prompt + "Question: " + question +"\n"+ " Answer: "

    # input_prompt = prompt + make_prompt(question, None)
    # print(input_prompt)
    output_text = inference(input_prompt)

    model_ans = extract_content(output_text, tag_type="answer")
    model_confidence = extract_content(output_text, tag_type="confidence")
    correct = 1 if any(answer.lower() in model_ans.lower() for answer in answers) else 0
    unsure = 1 if "unsure" in model_confidence.lower() else 0
    Refusal += unsure
    Q += (1-unsure)
    un_c_match = 1 if unsure and correct else 0
    
    CC, II = 0, 0
    if correct == 1 and unsure == 0:
        C += 1
        CC = 1
    if correct == 0 and unsure == 0:
        I += 1
        II = 1
    # ipdb.set_trace()
    rft_results.append({
        "question": question,
        "answers": answers,
        "model_ans": model_ans,
        "model_confidence": model_confidence,
        "correct": correct,
        "unsure": unsure,
        "un_c_match": un_c_match,
        "output_text": output_text,
        "I": II,
        "C": CC,
    })
    correct_sample += correct
    un_c_match_sample += un_c_match

print(f"total_sample: {total_sample}")
print(f"correct_sample: {correct_sample}")
print(f"accuracy: {correct_sample/total_sample}")

AED = math.sqrt((I * I + (total_sample - C) * (total_sample - C)) / (2 * total_sample * total_sample))
print(f"AED: {AED}")
print(f"Refusal: {Refusal}")
print(f"拒绝率: {Refusal/total_sample}")
rft_results.append({
    "total_sample": total_sample,
    "correct_sample": correct_sample,
    "accuracy": correct_sample/total_sample,
    "refusal": Refusal,
    'refusal_rate': Refusal/total_sample,
    'answer_sample': Q,
    'answer_rate': Q/total_sample,
    'un_c_match_sample': un_c_match_sample,
    'un_c_match_rate': un_c_match_sample/total_sample,
    'I': I,
    'C': C,
    "AED": AED
})

# 保存结果
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
json.dump(rft_results, open(args.output_path, "w"), indent=4)
print(f"结果已保存到: {args.output_path}")