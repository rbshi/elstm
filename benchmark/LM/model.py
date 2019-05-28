import torch.nn as nn
import numpy as np
from RNN.LSTM import LSTM 
from utils.Param_transfer import get_sparsity
import numpy as np

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn_type = rnn_type

        if rnn_type == 'LSTM':
            self.lstm_input = LSTM(ninp, nhid)
            self.lstm_hidden = LSTM(nhid, nhid)
        elif rnn_type == 'GRU':
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.hidden_sparsity = {}
        # self.hidden_states = {}
        for i in range(self.nlayers):
            #self.hidden_states['layer{}'.format(i)] = []
            self.hidden_sparsity['layer{}'.format(i)] = []
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, sparse=False, h_sp=[0.,0.], h_th=[0.,0.], block=-1):
        emb = self.drop(self.encoder(input))

        # # output vec_x for RISC-V simulation
        # emb_transpose = emb.transpose_(0,1)
        # emb_flat = emb_transpose.reshape(-1).numpy()
        # np.savez('wiki-x.npz', x=emb_flat)

        #output = emb
        if self.rnn_type == "LSTM":
            final_state = []
            for i in range(self.nlayers):
                if i == 0:
                    output, state = self.lstm_input(emb, hidden[0][i], hidden[1][i], sparse=sparse, h_sp=h_sp[0], h_th=h_th[0], block=block)

                    # # output vec_h for RISC-V simulation
                    # output_transpose = output.transpose_(0,1)
                    # h_flat = output_transpose.reshape(-1).numpy()
                    # np.savez('wiki-h.npz', x=h_flat)


                else:
                    output, state = self.lstm_hidden(output, hidden[0][i], hidden[1][i], sparse=sparse, h_sp=h_sp[1], h_th=h_th[1], block=block)
                if i != self.nlayers - 1:
                    output = self.drop(output)
                final_state.append(state)
                #self.hidden_states['layer{}'.format(i)].append(output.numpy())
                self.hidden_sparsity['layer{}'.format(i)].append(get_sparsity(output))
            final_state = tuple(final_state)
        else:
            output, final_state = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), final_state

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    # def save_hidden(self, filename):
    #     for k, v in self.hidden_states.items():
    #         layer_file = "{}_{}.npy".format(filename, k)
    #         with open(layer_file, 'ab') as f:
    #             np.save(f, np.asarray(v).flatten())
    #         self.hidden_states[k] = []