import torch
import torch.nn.functional as F
from copy import deepcopy

from torch import sigmoid 
from torch import tanh
import numpy as np
# propagate convolutional or linear layer
def propagate_conv_linear(relevant, irrelevant, module, device='cuda'):
    stabilizing_constant=1e-20
    bias = deepcopy(module)(torch.zeros(relevant.size()).to(device))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias

    # elementwise proportional
    prop_rel = torch.abs(rel)
    prop_irrel = torch.abs(irrel)
    prop_sum = prop_rel + prop_irrel +stabilizing_constant
    
    prop_rel = torch.div(prop_rel, prop_sum)
    prop_irrel = torch.div(prop_irrel, prop_sum)
    return rel + torch.mul(prop_rel, bias), irrel + torch.mul(prop_irrel, bias)


# propagate ReLu nonlinearity
def propagate_relu(relevant, irrelevant, activation, device='cuda'):
    both = (relevant + irrelevant).to(device)
    irrel_score = activation(relevant)
    total=activation(both)
    rel_score = total - irrel_score
    return rel_score,irrel_score 

# propagate Sigmoid nonlinearity
def propagate_sigmoid(relevant, irrelevant, activation, device='cuda'):
    return activation(relevant),activation(irrelevant)


# propagate maxpooling operation
def propagate_maxpooling(relevant, irrelevant, pooler,device='cuda'):
    pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices = True)
    unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
    # get both indices
    both, both_ind = pool(relevant + irrelevant)
    ones_out = torch.ones_like(both)
    size1 = relevant.size()
    mask_both = unpool(ones_out, both_ind, output_size=size1)
    # relevant
    rel = mask_both * relevant
    rel = pooler(rel)

    # irrelevant
    irrel = mask_both * irrelevant
    irrel = pooler(irrel)
    
    return rel, irrel

# propagate dropout operation
def propagate_dropout(relevant, irrelevant, dropout):
    return dropout(relevant), dropout(irrelevant)

    


