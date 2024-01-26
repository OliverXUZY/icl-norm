import torch

# Define the function to find subvector indices
def find_subvector_indices(inputs_right, inputs_sub):
    len_sub = len(inputs_sub)
    len_right = len(inputs_right)

    for i in range(len_right - len_sub + 1):
        if torch.equal(inputs_right[i:i + len_sub], inputs_sub):
            return i, i + len_sub - 1

    return -1, -1  # Return -1, -1 if subvector is not found

# Example tensors
inputs_right = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
inputs_sub = torch.tensor([4, 5, 6])

# Find indices
out = find_subvector_indices(inputs_right, inputs_sub)

print(out)
