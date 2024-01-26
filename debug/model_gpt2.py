
import torch
import random
import argparse
import os
from transformers import GPT2Model

import sys
sys.path.insert(0,"/Users/zyxu/Documents/py/nlp/zms")

def _add_hooks(model):
    # Assuming model is a GPT2 instance
    attention_matrices = []

    # Hook function to capture attention matrices
    def attention_hook(module, input, output):
        # output is a tuple where the first element is the attention matrix
        attention_matrices.append(output[1].clone())  # output[1] contains the attention matrix

    # Register the hook for each multi-head attention layer in each block
    for block in model.h:  # 'h' attribute contains the blocks in GPT-2 model
        block.attn.register_forward_hook(attention_hook)

    return attention_matrices


def main():

    model = GPT2Model.from_pretrained('gpt2')

    print(model)
    print("len(model.h): ", len(model.h))

    print(model.h[0])


if __name__ == "__main__":
    main()