import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
from RNN.LSTM import LSTM
from utils.Param_transfer import get_sparsity
import numpy as np

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_first=True):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.lstm1 = LSTM(input_size, hidden_size, batch_first)
        self.lstm2 = LSTM(hidden_size, hidden_size, batch_first)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.init_weights()
        # dict for hstate_sparsity record
        self.hstate_sp = {}

    def forward(self, x, states, sparse=False, h_sp=[0.,0.], h_th=[0.,0.], block=-1):

        final_state = []
        layer1, state = self.lstm1(x, states[0][0], states[1][0], sparse=sparse, h_sp=h_sp[0], h_th=h_th[0], block=block)
        final_state.append(state)

        # output vec_x for RISC-V simulation
        # f = open('mnist_h.txt', 'w')
        # h_flat = layer1.view(-1)
        # for elem in h_flat:
        #     f.write('{:f},'.format(elem))
        # f.close()

        # print (layer1.shape)
        # np.savetxt("layer1.csv", layer1[:, -1, :].numpy(), delimiter=',')
        # exit()

        self.hstate_sp['LSTM1'] = get_sparsity(layer1)
        layer2, state = self.lstm2(layer1, states[0][1], states[1][1], sparse=sparse, h_sp=h_sp[1], h_th=h_th[1], block=block)
        final_state.append(state)

        # np.savetxt("layer2.csv", layer2[:, -1, :].numpy(), delimiter=',')
        # exit()

        self.hstate_sp['LSTM2'] = get_sparsity(layer2)
        if self.batch_first:
            out = self.fc(layer2[:, -1, :])
        else:
            out = self.fc(layer2[-1, :, :])
        return out, tuple(final_state), self.hstate_sp

    def init_hidden(self, bsz):
        weight = next(self.parameters())
       
        return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
                weight.new_zeros(self.num_layers, bsz, self.hidden_size))
       
    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)