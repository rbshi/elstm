from __future__ import print_function, absolute_import
import os,sys 
sys.path.append(os.getcwd())
import argparse
import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
from model import RNN
from utils.Param_transfer import set_to_zero_sparsity, get_sparsity, set_to_zero_threshold
import numpy as np

sequence_length = 28 
input_size = 28
num_layers = 2
num_classes = 10
batch_size = 500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1111)

def main(arguments=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog="Sparse Evaluation on MNIST dataset")
    parser.add_argument('--nhid', type=int, default=128, help='number of hidden units per layer')
    parser.add_argument('-ws', '--w_sp', type=float, nargs='+', default=[0, 0, 0, 0], help="Weight sparsity setting.")
    parser.add_argument('-wt', '--w_th', type=float, default=0, help="Weight threshold setting.")
    parser.add_argument('-hs', '--h_sp', type=float, nargs='+', default=[0.,0.], help="Hidden state sparsity setting.")
    parser.add_argument('-ht', '--h_th', type=float, nargs='+', default=[0.,0.], help="Hidden state threshold setting.")
    parser.add_argument('-b', '--size_block', type=int, default=-1, help="Block size for hidden state sparsification.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose mode.")
    parser.add_argument('-model', '--model_path', default='MNIST/models/nhid:128-nlayer:2-epoch:10.ckpt', help="Model path.")

    args = parser.parse_args(arguments)

    # load model for GPU / CPU
    if torch.cuda.is_available():
        state_dict = torch.load(args.model_path)
    else:
        state_dict = torch.load(args.model_path, map_location = 'cpu')

    #sparsity_dict = {}

    for k, v in state_dict.items():
        if 'lstm1' in k:
            if 'weight_x' in k:
                state_dict[k] = set_to_zero_sparsity(v, sparsity=args.w_sp[0])
            if 'weight_h' in k:
                state_dict[k] = set_to_zero_sparsity(v, sparsity=args.w_sp[1])
        if 'lstm2' in k:
            if 'weight_x' in k:
                state_dict[k] = set_to_zero_sparsity(v, sparsity=args.w_sp[2])
            if 'weight_h' in k:
                state_dict[k] = set_to_zero_sparsity(v, sparsity=args.w_sp[3])

    # save weight and bias for RISCV simulation
    # l1_w = np.hstack((state_dict['lstm1.cell_f.weight_xf'].numpy(), state_dict['lstm1.cell_f.weight_xi'].numpy(), state_dict['lstm1.cell_f.weight_xu'].numpy(), state_dict['lstm1.cell_f.weight_xo'].numpy()))
    # l1_u = np.hstack((state_dict['lstm1.cell_f.weight_hf'].numpy(), state_dict['lstm1.cell_f.weight_hi'].numpy(), state_dict['lstm1.cell_f.weight_hu'].numpy(), state_dict['lstm1.cell_f.weight_ho'].numpy()))
    #
    # l2_w = np.hstack((state_dict['lstm2.cell_f.weight_xf'].numpy(), state_dict['lstm2.cell_f.weight_xi'].numpy(), state_dict['lstm2.cell_f.weight_xu'].numpy(), state_dict['lstm2.cell_f.weight_xo'].numpy()))
    # l2_u = np.hstack((state_dict['lstm2.cell_f.weight_hf'].numpy(), state_dict['lstm2.cell_f.weight_hi'].numpy(), state_dict['lstm2.cell_f.weight_hu'].numpy(), state_dict['lstm2.cell_f.weight_ho'].numpy()))
    #
    # l1_w = np.transpose(l1_w)
    # l1_u = np.transpose(l1_u)
    # l2_w = np.transpose(l2_w)
    # l2_u = np.transpose(l2_u)
    #
    # l1_b = np.hstack((state_dict['lstm1.cell_f.bias_f'].numpy(), state_dict['lstm1.cell_f.bias_i'].numpy(), state_dict['lstm1.cell_f.bias_u'].numpy(), state_dict['lstm1.cell_f.bias_o'].numpy()))
    # l2_b = np.hstack((state_dict['lstm2.cell_f.bias_f'].numpy(), state_dict['lstm2.cell_f.bias_i'].numpy(), state_dict['lstm2.cell_f.bias_u'].numpy(), state_dict['lstm2.cell_f.bias_o'].numpy()))
    #
    # np.savez('mnist-l1.npz', w=l1_w, u=l1_u, b=l1_b)
    # np.savez('mnist-l2.npz', w=l2_w, u=l2_u, b=l2_b)

    test_dataset = torchvision.datasets.MNIST(root='./MNIST/data/', train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    model = RNN(input_size, args.nhid, num_layers, num_classes)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.cuda()
        # model.half()
    # TODO: trans the model to half-precision float

    with torch.no_grad():
        correct = 0
        total = 0
        #hidden = model.init_hidden(batch_size)
        sparse_dict = {"LSTM1": 0., "LSTM2":0.}
        iteration = 0
        for images, labels in test_loader:

            # output vec_x for RISC-V simulation
            # f = open('mnist_x.txt', 'w')
            # images_flat = images.view(-1)
            # for elem in images_flat:
            #     f.write('{:f},'.format(elem))
            # f.close()

            hidden = (torch.zeros(num_layers, batch_size, args.nhid).to(device), torch.zeros(num_layers, batch_size, args.nhid).to(device))
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs, hidden, cur_dict = model(images, hidden, sparse=True, h_th=args.h_th, h_sp=args.h_sp, block=args.size_block)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            iteration += 1
            for k, v in sparse_dict.items():
                sparse_dict[k] += cur_dict[k]

        accuracy = 100.0 * correct / total
        print('|| Test Accuracy : {:.5f} || LSTM1 sparsity: {:.5f} || LSTM2 Sparsity: {:.5f} ||'.format(accuracy, sparse_dict['LSTM1']/iteration, sparse_dict['LSTM2']/iteration))


if __name__ == '__main__':
    main()