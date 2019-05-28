import torch
import torch.nn as nn
import math 


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
    
        self.activation = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, prev_h):
        output = torch.matmul(input, self.weight_ih) + self.bias_ih + torch.matmul(prev_h, self.weight_hh) + self.bias_hh
        output = self.activation(output)
        return output



class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VanillaRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = RNNCell(input_size, hidden_size)

    def forward(self, inputs, initial_state):
        time_steps = inputs.size(1)

        state = initial_state
        outputs = []
        for t in range(time_steps):
            state = self.cell(inputs[:, t, :], state)
            outputs.append(state)

        return torch.stack(outputs, 1)
