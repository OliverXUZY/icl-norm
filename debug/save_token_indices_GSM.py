from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

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
    model_path = "meta-llama/Llama-2-13b-hf"

    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    
    # Prepare the dataset
    cache_dir = "./data/cache"
    data_files = "./GSM-IC/GSM-IC_2step.json"
    dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir, split = "train")

    indices = []

    # Sampling and forward pass
    random.seed(42)  # Set the seed for reproducibility
    for _ in range(100):
        # Randomly sample an example
        ex = random.choice(dataset)
        
        # Define the substring you are looking for
        substring = ex['sentence_template'].format(role=ex['role'], number=ex['number'])

        # Find the index of the substring in 'new_question'
        inp_new_question = tokenizer(ex['new_question'], return_tensors='pt')['input_ids'].view(-1)
        inputs_sub = tokenizer(substring, return_tensors='pt')['input_ids'].view(-1)

        index = find_subvector_indices(inp_new_question, inputs_sub[1:])
        print(index)
    
        indices.append(index)
    
    # save_json(indices, "./data/indices.json")


if __name__ == "__main__":
    main()

    # At this point, attention_matrices contains the attention data for this sample

# The attention_matrices list now contains the attention matrices for each of the 100 samples
