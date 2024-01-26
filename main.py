from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from datasets import load_dataset
import torch
import random

# Define the hook function
def add_attention_hooks(model):
    attention_matrices = []

    def attention_hook(module, input, output):
        # print("len(output): ", len(output))
        attention_matrix = output[2]  # This gets the actual attention matrix
        print("len(attention_matrix): ", len(attention_matrix))
        print("attention_matrix.shape: ", attention_matrix.shape)
        attention_matrices.append(attention_matrix.clone())

    # for block in model.h:
    #     block.attn.register_forward_hook(attention_hook)
    block_id = len(model.h) - 1
    block = model.h[block_id]
    block.attn.register_forward_hook(attention_hook)

    return attention_matrices

def main():
    # Load GPT-2 model
    # Load the configuration of GPT-2
    config = GPT2Config.from_pretrained('gpt2')

    # Set output_attentions to True
    config.output_attentions = True

    # Reload the model with the updated configuration
    model = GPT2Model(config)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Add hooks to the model
    attention_matrices = add_attention_hooks(model)

    # Prepare the dataset
    cache_dir = "./data/cache"
    data_files = "./GSM-IC/GSM-IC_2step.json"
    dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir, split = "train")

    # Sampling and forward pass
    random.seed(42)  # Set the seed for reproducibility
    for _ in range(1):
        # Randomly sample an example
        ex = random.choice(dataset)
        

        # Define the substring you are looking for
        substring = ex['sentence_template'].format(role=ex['role'], number=ex['number'])

        # Find the index of the substring in 'new_question'
        start = ex['new_question'].find(substring)
        end = start + len(substring)

        ex['answer'] = "703"

        inp_right = ex['new_question'] + " " + ex['answer']

        wrong = "23" if ex['answer'] != "23" else "307"
        # wrong = ex['answer'] + "1"

        inp_wrong = ex['new_question'] + " " + wrong

        # Preprocess the text
        inputs_right = tokenizer(inp_right, return_tensors='pt')
        print(inp_right)
        print(inputs_right)

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(**inputs_right)
        
        inputs_wrong = tokenizer(inp_wrong, return_tensors='pt')
        print(inp_wrong)
        print(inputs_wrong)

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(**inputs_wrong)
    

    print(type(attention_matrices))

    print(len(attention_matrices))


    


if __name__ == "__main__":
    main()

    # At this point, attention_matrices contains the attention data for this sample

# The attention_matrices list now contains the attention matrices for each of the 100 samples
