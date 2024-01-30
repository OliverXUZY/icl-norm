from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import sys
sys.path.insert(0, "/Users/zyxu/Documents/py/NLP/icl-norm/")
from datasets import load_dataset
import torch
import random
from utils import save_json

def find_subvector_indices(inputs_right, inputs_sub):
    len_sub = len(inputs_sub)
    len_right = len(inputs_right)

    for i in range(len_right - len_sub + 1):
        if torch.equal(inputs_right[i:i + len_sub], inputs_sub):
            return i, i + len_sub - 1

    return -1, -1  # Return -1, -1 if subvector is not found

def main():
    
    # Prepare the dataset
    cache_dir = "./data/cache"
    data_files = "./GSM-IC/GSM-IC_2step.json"
    gsm = load_dataset("json", data_files=data_files, cache_dir=cache_dir, split = "train")

    
    sst2 = load_dataset("glue", "sst2", cache_dir=cache_dir, split = "train")
    # print(len(gsm))
    # print(len(sst2))
    # print(type(sst2[0]['label']))
    # assert False

    



    # print(sst2)

    # print(sst2[0])

    # assert False

    model_path = "meta-llama/Llama-2-13b-hf"

    tokenizer = LlamaTokenizer.from_pretrained(model_path)



    indices = []

    # Sampling and forward pass
    random.seed(42)  # Set the seed for reproducibility
    # Sample 100 random indices from 1 to 10000
    sample_indices = random.sample(range(1, 10001), 100)


    for idx in sample_indices:
        # Randomly sample an example
        ex = sst2[idx+1]
        irre = gsm[idx+1]

        print(ex)

        # Define the substring you are looking for
        substring = irre['sentence_template'].format(role=irre['role'], number=irre['number'])

        print(substring)

        print(ex['sentence'] + "\n" + substring)

        
        # print("===========")

        

        # Find the index of the substring in 'new_question'
        inp_ori_question = tokenizer(ex['sentence'], return_tensors='pt')['input_ids'].view(-1)
        print(inp_ori_question)
        # inputs_sub = tokenizer(substring, return_tensors='pt')['input_ids'].view(-1)

        # print(inputs_sub)

        

        inp_new_question = tokenizer(ex['sentence'] + "\n"+ substring, return_tensors='pt')['input_ids'].view(-1)
        print(inp_new_question)

        input_positive = f"{ex['sentence']}\n{substring}\nQuestion: Is this sentence positive or negative?\nAnswer: positive"
        input_positive = tokenizer(input_positive, return_tensors='pt')['input_ids'].view(-1)

        input_negative = f"{ex['sentence']}\n{substring}\nQuestion: Is this sentence positive or negative?\nAnswer: negative"
        input_negative = tokenizer(input_negative, return_tensors='pt')['input_ids'].view(-1)
        print("===")
        print(input_positive)
        print(input_negative)



        index = (len(inp_ori_question), len(inp_new_question))

        
    
        indices.append(index)
        # assert False
    
    save_json(indices, "./data/indices_sst2.json")


if __name__ == "__main__":
    main()

    # At this point, attention_matrices contains the attention data for this sample

# The attention_matrices list now contains the attention matrices for each of the 100 samples
