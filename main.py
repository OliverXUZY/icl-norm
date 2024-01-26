from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import argparse
from datasets import load_dataset
import torch
import random
from utils import save_json
# Check if CUDA is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Define the hook function

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=10)
    return parser.parse_args()


def add_attention_hooks(model):
    attention_matrices = []

    def attention_hook(module, input, output):
        # print("len(output): ", len(output))
        attention_matrix = output[1]  # This gets the actual attention matrix
        # print("attention_matrix", attention_matrix)
        # print("len(attention_matrix): ", len(attention_matrix))
        attention_matrix = attention_matrix.squeeze().mean(0)
        print("attention_matrix.shape: ", attention_matrix.shape)
        attention_matrices.append(attention_matrix.clone())

    # for block in model.model.layers:
    #     block.self_attn.register_forward_hook(attention_hook)
    block_id = len(model.model.layers) - 1
    block = model.model.layers[block_id]
    block.self_attn.register_forward_hook(attention_hook)

    return attention_matrices



def main():
    args = parse_args()
    model_path = args.model_id

    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    
    # Set output_attentions to True
    model = LlamaForCausalLM.from_pretrained(model_path,
                                attn_implementation="eager",
                                device_map='auto')
    model.config.output_attentions = True
    mofrl = model.to(device)
    
    # Prepare the dataset
    cache_dir = "./data/cache"
    data_files = "./GSM-IC/GSM-IC_2step.json"
    dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir, split = "train")

    # Add hooks to the model
    attention_matrices = add_attention_hooks(model)

    model.eval()
    # Sampling and forward pass
    random.seed(42)  # Set the seed for reproducibility
    for _ in range(args.batch_size):
        # Randomly sample an example
        ex = random.choice(dataset)
        
        # Define the substring you are looking for
        substring = ex['sentence_template'].format(role=ex['role'], number=ex['number'])

        inp_right = ex['new_question'] + " " + ex['answer']

        wrong = "23" if ex['answer'] != "23" else "307"
        inp_wrong = ex['new_question'] + " " + wrong

        # Preprocess the text
        inputs_right = tokenizer(inp_right, return_tensors='pt')
        print(inp_right)
        # print(inputs_right)
        print("length of tokens: ", inputs_right['input_ids'].shape)

        
        inputs_wrong = tokenizer(inp_wrong, return_tensors='pt')
        

        print(inp_wrong)
        # print(inputs_wrong)nvidia
        print("length of tokens: ", inputs_wrong['input_ids'].shape)

        # assert False
        # Forward pass through the model
        inputs_right = inputs_right.to(device)
        inputs_wrong = inputs_wrong.to(device)
        with torch.no_grad():
            outputs = model(**inputs_right)
        # Forward pass through the model
        with torch.no_grad():
            outputs = model(**inputs_wrong)
    

    print(type(attention_matrices))

    print(len(attention_matrices))

    

    # Saving the list of tensors to a file
    save_name = model_path.split("/")[1]
    torch.save(attention_matrices, f'save/attn_mtx_{save_name}.pth')

    pos = []
    neg = []
    i = 0
    while i < len(attn_matrix)-1:
        pos.append(attn_matrix[i])
        neg.append(attn_matrix[i+1])
        i += 2
    
    norms = []
    for i, (pmat, nmat) in enumerate(zip(pos,neg)):
        tmp = []
        start, end = indices[i]
        tmp.append(pmat[-2, :start].norm())     # positive relevent
        tmp.append(pmat[-2, start:end].norm())  # positive irre
        tmp.append(nmat[-2, :start].norm())     # neg relevent
        tmp.append(nmat[-2, start:end].norm())  # neg irre

        norms.append(tmp)

    
    norms = np.array(f"save/norms_{save_name}.pth", norms)


    


if __name__ == "__main__":
    main()

    # At this point, attention_matrices contains the attention data for this sample

# The attention_matrices list now contains the attention matrices for each of the 100 samples
