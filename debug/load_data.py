import os
import sys
sys.path.insert(0,"/Users/zyxu/Documents/py/NLP/zms")
import argparse
import random
import torch

import json

from datasets import load_dataset

cache_dir = "./data/cache"
data_files = "./GSM-IC/GSM-IC_2step.json"
dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir, split = "train")

# print(dataset)
random.seed(42)  # Set the seed for reproducibility
ex = random.choice(dataset)
# ex = dataset[9]

# print(ex)
# print(ex.keys())
irre = ex['sentence_template'].format(role=ex['role'], number=ex['number'])
print("irre: ", irre)

print("new question: ", ex['new_question'])
print("length: ", len(ex['new_question']))
index_substring = ex['new_question'].find(irre)


length_new_question = len(ex['new_question'])

print("index_substring: ", index_substring)






wrong = ex['answer'][::-1] if len(ex['answer'])>1 else ex['answer'] + "1"
print("answr: ", ex['answer'])
print("wrong: ", wrong)


print(ex['new_question'] + " " + wrong)
# for key,val in ex.items():
#     print(key)
#     print(val)
#     print("===")



