from __future__ import print_function, absolute_import
import os,sys 
sys.path.append(os.getcwd())
import argparse
import math

import torch
import torch.nn as nn
import torch.onnx
import data
import model
import numpy as np
#from model import RNNModel
from utils.Param_transfer import set_to_zero_sparsity, set_to_zero_threshold
#from config import SmallConfig, MediumConfig, LargeConfig


RNN_type = "LSTM"
nlayers = 2 
dropout = 0.65
bptt = 35
tied = False
eval_batch_size = 50
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(model, data_source, ntokens, h_sp=[0.,0.], h_th=[0.,0.], block=-1):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    #ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden, sparse=True, h_sp=h_sp, h_th=h_th, block=block)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)
 

def main(arguments=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog="LM_eval", description="Evaluate network with different sparsity")

    parser.add_argument('--data', type=str, default='./LM/data/PTB/', help='location of the data corpus')
    parser.add_argument('-m', '--mode', required=True, choices=['sparsity', 'threshold'], help="Sparse mode selection: Use sparsity setting / threshold setting.")
    parser.add_argument('-ws', '--w_sp', type=float, nargs='+', default=[0,0,0,0], help="Weight sparsity setting.")
    parser.add_argument('--nhid', type=int, default=800, help='number of hidden units per layer')
    parser.add_argument('--emsize', type=int, default=800, help='size of word embeddings')
    parser.add_argument('-wt', '--w_th', type=float, default=0, help="Weight threshold setting.")
    parser.add_argument('-hs', '--h_sp', type=float, nargs='+', default=[0.,0.], help="Hidden state sparsity setting.")
    parser.add_argument('-ht', '--h_th', type=float, nargs='+', default=[0.,0.], help="Hidden state threshold setting.")
    parser.add_argument('-b', '--size_block', type=int, default=-1, help="Block size for hidden state sparsification.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose mode.")
    parser.add_argument('--model_path', default='./LM/models/PTB/model:LSTM-em:1500-nhid:1500-nlayers:2-bptt:35-epoch:15-lr:20-tied:False-l1:False-l1_lambda:1e-05-dropout:0.65.ckpt', help="The path to saved model")
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args(arguments)

    state_dict = torch.load(args.model_path, map_location = lambda storage, loc:storage)

    torch.manual_seed(args.seed)

    corpus = data.Corpus(args.data)
    test_data = batchify(corpus.test, eval_batch_size)

    # weight sparsification
    if args.mode == 'sparsity':
        for k, v in state_dict.items():
            if 'lstm_input' in k:
                if 'weight_x' in k:
                    state_dict[k] = set_to_zero_sparsity(v, sparsity=args.w_sp[0])
                if 'weight_h' in k:
                    state_dict[k] = set_to_zero_sparsity(v, sparsity=args.w_sp[1])
            if 'lstm_hidden' in k:
                if 'weight_x' in k:
                    state_dict[k] = set_to_zero_sparsity(v, sparsity=args.w_sp[2])
                if 'weight_h' in k:
                    state_dict[k] = set_to_zero_sparsity(v, sparsity=args.w_sp[3])

    else:
        for k, v in state_dict.items():
            if 'lstm' in k:
                state_dict[k] = set_to_zero_threshold(v, threshold=args.w_th)

    # # save weight and bias for RISCV simulation
    # l1_w = np.hstack((state_dict['lstm_input.cell_f.weight_xf'].numpy(), state_dict['lstm_input.cell_f.weight_xi'].numpy(), state_dict['lstm_input.cell_f.weight_xu'].numpy(), state_dict['lstm_input.cell_f.weight_xo'].numpy()))
    # l1_u = np.hstack((state_dict['lstm_input.cell_f.weight_hf'].numpy(), state_dict['lstm_input.cell_f.weight_hi'].numpy(), state_dict['lstm_input.cell_f.weight_hu'].numpy(), state_dict['lstm_input.cell_f.weight_ho'].numpy()))
    #
    # l2_w = np.hstack((state_dict['lstm_hidden.cell_f.weight_xf'].numpy(), state_dict['lstm_hidden.cell_f.weight_xi'].numpy(), state_dict['lstm_hidden.cell_f.weight_xu'].numpy(), state_dict['lstm_hidden.cell_f.weight_xo'].numpy()))
    # l2_u = np.hstack((state_dict['lstm_hidden.cell_f.weight_hf'].numpy(), state_dict['lstm_hidden.cell_f.weight_hi'].numpy(), state_dict['lstm_hidden.cell_f.weight_hu'].numpy(), state_dict['lstm_hidden.cell_f.weight_ho'].numpy()))
    #
    # l1_w = np.transpose(l1_w)
    # l1_u = np.transpose(l1_u)
    # l2_w = np.transpose(l2_w)
    # l2_u = np.transpose(l2_u)
    #
    # l1_b = np.hstack((state_dict['lstm_input.cell_f.bias_f'].numpy(), state_dict['lstm_input.cell_f.bias_i'].numpy(), state_dict['lstm_input.cell_f.bias_u'].numpy(), state_dict['lstm_input.cell_f.bias_o'].numpy()))
    # l2_b = np.hstack((state_dict['lstm_hidden.cell_f.bias_f'].numpy(), state_dict['lstm_hidden.cell_f.bias_i'].numpy(), state_dict['lstm_hidden.cell_f.bias_u'].numpy(), state_dict['lstm_hidden.cell_f.bias_o'].numpy()))
    #
    # np.savez('ptb-l1.npz', w=l1_w, u=l1_u, b=l1_b)
    # np.savez('ptb-l2.npz', w=l2_w, u=l2_u, b=l2_b)




    
    ntokens = len(corpus.dictionary)
    test_model = model.RNNModel(RNN_type, ntokens, args.emsize, args.nhid, nlayers, dropout, tied).to(device)
    test_model.load_state_dict(state_dict)
    test_loss = evaluate(test_model, test_data, ntokens, h_th=args.h_th, block=args.size_block)
    print("|| Evaluate on test data || loss: {:5.2f} || PPL {:8.2f} ||".format(
        test_loss, math.exp(test_loss)
    ))
    for k, v in test_model.hidden_sparsity.items():
        print("|{} | num_batch: {}| sparsity {:5.4f}|".format(k, len(v), sum(v)/float(len(v))))
   


if __name__ == '__main__':
    main()