import json
import ipdb
from vllm import LLM, SamplingParams
import json
import os
model_name = "/data/ah/code/rl/open-rs/results/pararel"
model_name = "/data/ah/code/rl/open-rs/results/qwen1.5b_pararel_origin_reward_fix_unreward/checkpoint-600"
model_name = "/data/ah/code/rl/open-rs/results/llama3/pararel_batch10_51_2"
model_name = "/data/ah/code/rl/open-rs/results/llama3/pararel_batch10"
model_name = "results/llama3/pararel_batch10_53_strict_contents_format8"
model_name = "results/llama3/pararel_batch10_53_strict_answer_se_1"
model_name = "results/llama3/pararel_batch10_53_strict_answer_se_1_un6"
# cuda :1
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
model = LLM(
    model=model_name,
)
print("model load sucess")

# from src.open_r1.rewards_u import extract_content, are_equivalent
from tqdm import tqdm
import os
import re
import math

# 定义采样参数
sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=40
)

def extract_content(text, tag_type="answer"):
    if tag_type == "answer":
        pattern = r"<answer>(.*?)</answer>"
    else:  # confidence
        pattern = r"<confidence>(.*?)</confidence>"
        # confidence = text.split("<confidence>")[-1]
        # confidence = confidence.split("</confidence>")[0]
        # return confidence.strip()

    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else ""

data_path = "dataset/ID_test_pararel.json"
# data_path = "dataset/OOD_test_pararel.json"
# data_path = "dataset/pararel_training.json"
dataset = json.load(open(data_path))

def inference(input_text):
    outputs = model.generate(
            input_text,
            sampling_params=sampling_params
    )
    
    output_text = outputs[0].outputs[0].text
    # 找到第一个</confidence>， 保留第一个</confidence>及其之前的文本
    result = output_text.split('</confidence>')[0] + '</confidence>'
    # print(result)
    return result

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

rft_pararel_results = []
total_sample = len(dataset)
correct_sample = 0
un_c_match_sample = 0
Q, I, C = 0, 0, 0
for data in tqdm(dataset):
    question = data["question"]
    answer = data["answer"]
    prompt = sys_prompt + "Question: " + question +"\n"+ " Answer: "
    output_text = inference(prompt)
    model_ans = extract_content(output_text, "answer")
    model_confidence = extract_content(output_text, "confidence")
    # lines = output_text.split('\n')
    # if len(lines) == 0:
    #     lines.append("")
    # if len(lines) == 1:
    #     lines.append("")
    # model_ans = lines[0].removeprefix('<answer>').removesuffix('</answer>')
    # model_confidence = lines[1]
    # strict_eq = are_equivalent(answer, model_ans, strict_entailment=True, example=None)
    # correct = 1 if answer.lower() in model_ans.lower() or strict_eq else 0
    correct = 1 if answer.lower() in model_ans.lower() else 0
    unsure = 1 if "unsure" in model_confidence.lower() else 0
    un_c_match = 1 if ("unsure" in model_confidence.lower() and correct == 0 ) or ("sure" in model_confidence.lower() and correct == 1) else 0
    # ipdb.set_trace()
    CC, II = 0, 0
    if correct == 1 and unsure == 0:
        C += 1
        CC = 1
    if correct == 0 and unsure == 0:
        I += 1
        II = 1
    # ipdb.set_trace()
    rft_pararel_results.append({
        "question": question,
        "answer": answer,
        "model_ans": model_ans,
        "model_confidence": model_confidence,
        # "strict_eq": strict_eq,
        "correct": correct,
        "unsure": unsure,
        "un_c_match": un_c_match,
        "output_text": output_text,
        "I": II,
        "C": CC,
    })
    correct_sample += correct
    un_c_match_sample += un_c_match
D = total_sample
AED = math.sqrt(   (I * I + (D - C) * (D - C)) / (2 * D * D)   )

print(f"total_sample: {total_sample}")
print(f"correct_sample: {correct_sample}")
print(f"un_c_match_sample: {un_c_match_sample}")
print(f"accuracy: {correct_sample/total_sample}")
print(f"un_c_match_accuracy: {un_c_match_sample/total_sample}")
print(f"AED: {AED}")
rft_pararel_results.append({
    "total_sample": total_sample,
    "correct_sample": correct_sample,
    "accuracy": correct_sample/total_sample,
    "un_c_match_sample": un_c_match_sample,
    "un_c_match_accuracy": un_c_match_sample/total_sample,
    "AED": AED,
})
# save results
os.makedirs("results/eval_pararel/llama3/results/llama3/pararel_batch10_53_strict_answer_se_1_un6", exist_ok=True)
json.dump(rft_pararel_results, open("results/eval_pararel/llama3/results/llama3/pararel_batch10_53_strict_answer_se_1_un6/ID_pararel.json", "w"), indent=4)
