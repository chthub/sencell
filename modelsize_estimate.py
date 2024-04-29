import torch
import torch.nn as nn
import numpy as np

# Define a function to calculate model size
def modelsize(model, input, type_size=4):
    """
    Calculates and prints the size of a PyTorch model in terms of parameter memory and intermediate variable memory.

    Args:
    model (nn.Module): The PyTorch model to analyze.
    input (torch.Tensor): An example input tensor to feed through the model for size analysis.
    type_size (int): Size in bytes of each model parameter or intermediate variable's data type (default: 4 for float32).

    Returns:
    None, but prints model information including total parameter memory and intermediate variable memory.
    """

    # Calculate the total number of parameters
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    # Prepare a clone of the input tensor
    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    # Traverse all modules in the model to calculate output sizes
    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        # Skip in-place ReLU to prevent modifying the input_ tensor
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        # Forward the input through each module and record output size
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    # Calculate the total number of intermediate variables
    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    # Print information about intermediate variables without and with backward computations
    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))

