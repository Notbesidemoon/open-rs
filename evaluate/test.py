import json
import ipdb
from vllm import LLM, SamplingParams
import json
import os
import math
import argparse



from src.open_r1.rewards_u import extract_content, are_equivalent
from tqdm import tqdm
import os
# 定义采样参数
sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=40
)

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
    print(result)
    return result

def main(args):
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    model = LLM(
        model=args.model_name,
    )
    output_json = {'samples': [], 'stats': {}}
    total_sample = len(dataset)
    correct_sample = 0
    un_c_match_sample = 0
    Q, I, C = 0, 0, 0

    for data in tqdm(dataset):
        question = data["question"]
        answer = data["answer"]
        prompt = sys_prompt + "Question: " + question +"\n"+ " Answer: "
        # input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:1")
        # outputs = model.generate(input_ids, max_new_tokens=100)
        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(response)
        # print('-'*100)
        output_text = inference(prompt)
        # model_ans = extract_content(output_text, "answer")
        # model_confidence = extract_content(output_text, "confidence")
        lines = output_text.split('\n')
        if len(lines) == 0:
            lines.append("")
        if len(lines) == 1:
            lines.append("")
        model_ans = lines[0].removeprefix('<answer>').removesuffix('</answer>')
        model_confidence = lines[1]
        strict_eq = are_equivalent(answer, model_ans, strict_entailment=True, example=None)
        correct = 1 if answer.lower() in model_ans.lower() or strict_eq else 0
        unsure = 1 if "unsure" in model_confidence.lower() else 0
        un_c_match = 1 if ("unsure" in model_confidence.lower() and correct == 0 ) or ("sure" in model_confidence.lower() and correct == 1) else 0
        # ipdb.set_trace()
        if not unsure:
            Q += 1
        if correct == 0:
            I += 1
        else:
            C += 1
        output_json['samples'].append({
            "question": question,
            "answer": answer,
            "model_ans": model_ans,
            "model_confidence": model_confidence,
            "strict_eq": strict_eq,
            "correct": correct,
            "unsure": unsure,
            "un_c_match": un_c_match,
            "output_text": output_text
        })
        correct_sample += correct
        un_c_match_sample += un_c_match

    print(f"total_sample: {total_sample}")
    print(f"correct_sample: {correct_sample}")
    print(f"un_c_match_sample: {un_c_match_sample}")
    print(f"accuracy: {correct_sample/total_sample}")
    print(f"un_c_match_accuracy: {un_c_match_sample/total_sample}")

    false
    AED = math.sqrt(   (I * I + (total_sample - C) * (total_sample - C)) / (2 * total_sample * total_sample)   )
    output_json['stats'] = {
        "total_sample": total_sample,
        "correct_sample": correct_sample,
        "accuracy": correct_sample/total_sample,
        "un_c_match_sample": un_c_match_sample,
        "un_c_match_accuracy": un_c_match_sample/total_sample,
        "AED": AED
    }
    # save results
    os.makedirs("results/eval_pararel", exist_ok=True)
    json.dump(output_json, open(args.output_path, "w"), indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=model_name)
    parser.add_argument("--cuda", type=str, default="1")
    parser.add_argument("--data_path", type=str, default="data/test.json")
    parser.add_argument("--output_path", type=str, default="results/eval_pararel/ID_rft_pararel_results_se_eq.json")
    parser.add_argument("--sys_prompt", type=str, default=sys_prompt)
    args = parser.parse_args()
    main(args)
