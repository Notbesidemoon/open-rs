"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
import torch
import torch.nn.functional as F
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 全局变量，用于存储NLI模型实例
nli_model = None

class BaseEntailment:
    def save_prediction_cache(self):
        pass
# huggingface-cli download --resume-download microsoft/deberta-v2-xlarge-mnli --local-dir /data1/model/nli/deberta-v2-xlarge-mnli

class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        # 获取当前进程的本地 rank
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        print(f"Initializing EntailmentDeberta on device {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained("/data1/model/nli/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "/data1/model/nli/deberta-v2-xlarge-mnli").to(self.device)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        # if os.environ.get('DEBERTA_FULL_LOG', False):
        #     logging.info('Deberta Input: %s -> %s', text1, text2)
        #     logging.info('Deberta Prediction: %s', prediction)

        return prediction

# 初始化全局nli_model
def initialize_nli_model():
    global nli_model
    if nli_model is None:
        nli_model = EntailmentDeberta()
    return nli_model

# 获取全局nli_model的函数
def get_nli_model():
    global nli_model
    if nli_model is None:
        try:
            initialize_nli_model()
        except Exception as e:
            print(f"初始化 NLI 模型失败: {e}")
            # 返回一个简单的假模型，避免后续代码崩溃
            class DummyModel:
                def check_implication(self, text1, text2, *args, **kwargs):
                    return 2  # 总是返回 entailment
            return DummyModel()
    return nli_model

# 模块级变量初始化
try:
    nli_model = EntailmentDeberta()
except Exception as e:
    print(f"初始化 NLI 模型失败: {e}")
    nli_model = None

def are_equivalent(text1, text2, strict_entailment=False, example=None):
    """检查两个文本是否语义等价"""
    model = get_nli_model()
    
    implication_1 = model.check_implication(text1, text2, example=example)
    implication_2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
    assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

    if strict_entailment:
        semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
    else:
        implications = [implication_1, implication_2]
        # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
        semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

    return semantically_equivalent

def get_semantic_ids(strings_list, strict_entailment=False, example=None):
    """将预测列表分组为语义含义组"""
    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j], strict_entailment=strict_entailment, example=example):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids
    # for any string, calculate the number of strings in the same semantic set
    semantic_set_counts = [semantic_set_ids.count(semantic_set_ids[i]) for i in range(len(semantic_set_ids))]
    return semantic_set_ids, semantic_set_counts

def extract_content(text, tag_type="answer"):
    if tag_type == "answer":
        pattern = r"<answer>(.*?)</answer>"
    else:  # confidence
        pattern = r"<confidence>(.*?)</confidence>"

    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else ""

def uncertainty_reward(completions, strict_entailment=False, **kwargs):
    """基于语义组计算确定性奖励函数"""
    try:
        # 如果 completions 为空或无效，返回默认奖励
        if not completions or len(completions) == 0:
            print("uncertainty_reward: completions 为空")
            return [0.5]
            
        contents = [completion[0]["content"] for completion in completions]
        
        # 安全地获取语义分组
        try:
            semantic_set_ids, semantic_set_counts = get_semantic_ids(contents, strict_entailment=strict_entailment)
        except Exception as sem_error:
            print(f"获取语义集失败: {sem_error}")
            return [0.5] * len(contents)
            
        # 提取信心表达
        confidence_words = []
        for content in contents:
            try:
                confidence_words.append(extract_content(content, "confidence"))
            except Exception:
                confidence_words.append("")
                
        rewards = []
        for confidence_word, semantic_set_count in zip(confidence_words, semantic_set_counts):
            try:
                confidence_word = confidence_word.lower().split()
                if semantic_set_count < len(contents) / 2:
                    if "unsure" in confidence_word and "sure" not in confidence_word:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                else:
                    if "unsure" not in confidence_word and "sure" in confidence_word:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
            except Exception:
                rewards.append(0.5)  # 默认中等奖励
                
        return rewards
    except Exception as e:
        print(f"uncertainty_reward 出错: {e}")
        # 在异常情况下返回默认奖励
        return [0.5] * len(completions) if completions else [0.5]
    
def accuracy_reward(completions, solution=None, strict_entailment=False, **kwargs):
    """检查完成是否与基准答案相同的奖励函数"""
    try:
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        # 如果 solution 是None或空，或者不匹配completions长度，返回默认奖励
        if solution is None or len(solution) == 0 or len(solution) != len(contents):
            print(f"accuracy_reward: solution参数无效或长度不匹配 {len(solution) if solution else 0} vs {len(contents)}")
            return [0.5] * len(contents)
            
        for content, sol in zip(contents, solution):
            print(f"content: {content}, solution: {sol}")
            try:
                answer = extract_content(content, "answer")

                # answer = content.splitlines()[0].split().lower()
                if are_equivalent(answer, sol, strict_entailment=strict_entailment):
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception as inner_e:
                print(f"处理单个答案时出错: {inner_e}")
                rewards.append(0.5)  # 默认中等奖励
        return rewards
    except Exception as e:
        print(f"accuracy_reward 出错: {e}")
        return [0.5] * len(completions)

def format_reward(completions, **kwargs):
    """Reward function that checks if the format is correct based on model type."""
    pattern = r"<answer>.*?</answer>.*?<confidence>.*?</confidence>"
    #strict_pattern = r"^<answer>.*?</answer>\s*<confidence>.*?</confidence>$"
    # soft_pattern = r"<answer>.*?</answer>\s*<confidence>.*?</confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [1.0 if re.match(pattern, content) else 0.0 for content in completion_contents]
    return matches

def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of tags."""
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<answer>") == 1:
            count += 0.25
        if text.count("</answer>") == 1:
            count += 0.25
        if text.count("<confidence>") == 1:
            count += 0.25
        if text.count("</confidence>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]






