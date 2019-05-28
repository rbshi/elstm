import math
import torch 
import torch.nn as nn
from utils.Param_transfer import set_to_zero_sparsity, set_to_zero_threshold


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input gate
        self.weight_xi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # forget gate
        self.weight_xf = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # update 
        self.weight_xu = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hu = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # ouput gate
        self.weight_xo = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        self.bias_i = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_f = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_o = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_u = nn.Parameter(torch.Tensor(hidden_size))

        self.activation = nn.Sigmoid()

        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

        # Orthogonal initialization 
        # self.bias_f.data.fill_(0.)
        # self.bias_i.data.fill_(0.)
        # self.bias_o.data.fill_(0.)
        # self.bias_u.data.fill_(0.)
        # nn.init.orthogonal_(self.weight_xi)
        # nn.init.orthogonal_(self.weight_hi)
        # nn.init.orthogonal_(self.weight_xf)
        # nn.init.orthogonal_(self.weight_hf)
        # nn.init.orthogonal_(self.weight_xu)
        # nn.init.orthogonal_(self.weight_hu)
        # nn.init.orthogonal_(self.weight_xo)
        # nn.init.orthogonal_(self.weight_ho)



    def forward(self, input, prev_h, prev_c):

        # if torch.cuda.is_available():
        #     input = input.type(torch.cuda.HalfTensor)
        #     prev_h = prev_h.type(torch.cuda.HalfTensor)
        #     prev_c = prev_c.type(torch.cuda.HalfTensor)

        input_gate = torch.matmul(input, self.weight_xi) + torch.matmul(prev_h, self.weight_hi) + self.bias_i 
        input_gate = self.activation(input_gate)

        forget_gate = torch.matmul(input, self.weight_xf) + torch.matmul(prev_h, self.weight_hf) + self.bias_f
        forget_gate = self.activation(forget_gate)

        output_gate = torch.matmul(input, self.weight_xo) + torch.matmul(prev_h, self.weight_ho) + self.bias_o
        output_gate = self.activation(output_gate) 

        update = torch.matmul(input, self.weight_xu) + torch.matmul(prev_h, self.weight_hu) + self.bias_u
        update = self.tanh(update)

        current_c = forget_gate * prev_c +  input_gate * update
        current_h = output_gate * self.tanh(current_c)     

        return current_h, current_c


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, batch_first=False, bidirectional=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.h_sp = h_sp
        self.time_major = not batch_first
        self.cell_f = LSTMCell(input_size, hidden_size)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.cell_b = LSTMCell(input_size, hidden_size)


    # def reset_parameters(self):
    #     std = 1.0 / math.sqrt(self.hidden_size)
    #     for w in self.parameters():
    #         w.data.uniform_(-std, std)

    def forward(self, inputs, initial_h, initial_c, sparse=False, h_sp=0., h_th=0., block=-1):
        # Input need to have size [batch_size, time_step, input_size]
        if self.time_major:
            time_steps = inputs.size(0)
        else:
            time_steps = inputs.size(1)
        

        if not self.bidirectional:
            h_f = initial_h
            c_f = initial_c 
        else:
            h_f = initial_h[0]
            c_f = initial_c[0]
            h_b = initial_h[1]
            c_b = initial_c[1]

        outputs_f = []
        if self.bidirectional:
            outputs_b = [] 

        for t in range(time_steps):
            if self.time_major:
                h_f, c_f = self.cell_f(inputs[t, :, :], h_f, c_f)
                if self.bidirectional:
                    h_b, c_b = self.cell_b(inputs[time_steps-1-t, :, :], h_b, c_b)
            else:
                h_f, c_f = self.cell_f(inputs[:, t, :], h_f, c_f)
                if self.bidirectional:
                    h_b, c_b = self.cell_b(inputs[:, time_steps-1-t, :], h_b, c_b)
            
            # Get sparse h
            if sparse:
                if h_sp > 0.:
                    h_f = set_to_zero_sparsity(h_f, sparsity=h_sp)
                    if self.bidirectional:
                        h_b = set_to_zero_sparsity(h_b, sparsity=h_sp)
                elif h_th > 0.:
                    h_f = set_to_zero_threshold(h_f, threshold=h_th, block=block)
                    if self.bidirectional:
                        h_b = set_to_zero_threshold(h_b, threshold=h_th, block=block)
            
            outputs_f.append(h_f)
            if self.bidirectional:
                outputs_b.append(h_b)

        if self.time_major:
            outputs_f = torch.stack(outputs_f)
            if self.bidirectional:
                outputs_b = torch.stack(outputs_b)
        else:
            outputs_f = torch.stack(outputs_f, 1)
            if self.bidirectional:
                outputs_b = torch.stack(outputs_b, 1)
        
        if self.bidirectional:
            outputs = torch.cat((outputs_f, outputs_b), 2)
            out_h = torch.stack([h_f, h_b], 0)
            out_c = torch.stack([c_f, c_b], 0)
        else:
            outputs = outputs_f
            out_h = h_f
            out_c = c_f 

        return outputs, (out_h, out_c)