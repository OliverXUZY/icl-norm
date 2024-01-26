import os
import sys
sys.path.insert(0,"/Users/zyxu/Documents/py/NLP/zms")
import argparse
import random
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

import json

from datasets import load_dataset


def find_subvector_indices(inputs_right, inputs_sub):
    len_sub = len(inputs_sub)
    len_right = len(inputs_right)

    for i in range(len_right - len_sub + 1):
        if torch.equal(inputs_right[i:i + len_sub], inputs_sub):
            return i, i + len_sub - 1

    return -1, -1  # Return -1, -1 if subvector is not found



cache_dir = "./data/cache"
data_files = "./GSM-IC/GSM-IC_2step.json"
dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir, split = "train")

# print(dataset)
random.seed(42)  # Set the seed for reproducibility
ex = random.choice(dataset)
# ex = dataset[9]

# print(ex)
print(ex.keys())
irre = ex['sentence_template'].format(role=ex['role'], number=ex['number'])
# print("irre: ", irre)

# print("new question: ", ex['new_question'])
# print("length: ", len(ex['new_question']))
index_substring = ex['new_question'].find(irre)


length_new_question = len(ex['new_question'])

# print("index_substring: ", index_substring)


wrong = "23" if ex['answer'] != "23" else "307"
# print("answr: ", ex['answer'])
prompt1 = ex['new_question'] + " " + ex['answer']
# print(prompt1)

# print("wrong: ", wrong)
prompt2 = ex['new_question'] + " " + wrong
# print(prompt2)



model_path = "meta-llama/Llama-2-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_path)
# Preprocess the text
print(prompt1)
inputs_right = tokenizer(prompt1, return_tensors='pt')['input_ids'].view(-1)
print(inputs_right.shape)
print(inputs_right)


# Preprocess the text
inputs_ori = tokenizer(ex['original_question'], return_tensors='pt')['input_ids'].view(-1)
print(ex['original_question'])
print(inputs_ori)
print(inputs_ori.shape)

# Preprocess the text
substring = ex['sentence_template'].format(role=ex['role'], number=ex['number'])
inputs_sub = tokenizer(substring, return_tensors='pt')['input_ids'].view(-1)
print(substring)
print(inputs_sub)
print(inputs_sub.shape)



print(find_subvector_indices(inputs_right, inputs_sub[1:]))



