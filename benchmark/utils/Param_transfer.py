import torch
import math 
import numpy as np 


def set_to_zero_sparsity(inputs, sparsity=0.):
    """
    args: 
    - inputs: the Tensor object of model weight 
    - sparsity: The sparsity we need 

    return:
    - the tensor with given sparsity
    """
    flatten = inputs.view(-1)
    absolute = torch.abs(flatten)
    sort = torch.sort(absolute)
    size = sort[0].size(0)
    threshold_index = int(math.floor(size * sparsity))
    threshold_value = sort[0][threshold_index]

    mask = torch.abs(inputs) > threshold_value
    mask = mask.float()
    return inputs * mask


def set_to_zero_threshold(inputs, threshold=0., block=-1):
    """
    args: 
    - inputs: the Tensor object of model weight 
    - threshold: The threshold for zero
    - block: block size (values are set to zero in a unit of block)
    return:
    - the tensor with given sparsity
    """

    alpha = 2

    if block < 0:
        absolute = torch.abs(inputs)
        mask = absolute > threshold
        mask = mask.float()
        return inputs * mask
    else:
        in_shape_batch = inputs.shape[0]
        in_shape_len = inputs.shape[1]
        in_reshape = inputs.view((in_shape_batch, int(in_shape_len/block), block))
        sum_block = torch.sum(torch.abs(in_reshape), dim=2)
        mask = ((sum_block >= threshold) | (torch.sum((torch.abs(in_reshape) > (threshold / alpha)), dim=2)!=0)).view(in_shape_batch,int(in_shape_len/block),1).repeat(1,1,block).view(in_shape_batch, in_shape_len)
        # mask = ((sum_block >= threshold)).view(in_shape_batch,int(in_shape_len/block),1).repeat(1,1,block).view(in_shape_batch, in_shape_len)
        return inputs * mask.float()

        # # Blocked
        # row_number = inputs.shape[0]
        # input_tensor_size = inputs.shape[1]
        # # assert(input_tensor_size % block == 0, "The hidden size should be an integer multiple of block size!")
        # for row in range(row_number):
        #     for i in range(0, input_tensor_size, block):
        #         sum_block = torch.sum(torch.abs(inputs[row][i:block]))
        #         if (sum_block < threshold) & (torch.sum(torch.abs(inputs[row][i:block]) > (threshold * alpha))==0):
        #             inputs[row][i:block] = 0.
        # return inputs


def get_sparsity(inputs):
    """
    args: 
    - inputs: inputs tensor object 

    return:
        the sparsity of inputs
    """
    flatten = inputs.view(-1)
    absolute = torch.abs(flatten)
    total = absolute.size()[0]
    mask = absolute > 0
    non_zero = torch.sum(mask)
    return 1. - int(non_zero)/total

