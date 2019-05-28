import argparse
import math

parser = argparse.ArgumentParser(description='Parameter ngrp selection in elstm.')
parser.add_argument('--xsize', type=int, default=800, help='size of input x in the LSTM layer')
parser.add_argument('--hsize', type=int, default=800, help='size of hidden state in the LSTM layer')
parser.add_argument('--ts', type=int, default=35, help='size of hidden state in the LSTM layer')
parser.add_argument('--npe', type=int, default=3, help='number of PE for Wx computation')
parser.add_argument('--spw', type=float, default=0.2, help='sparsity of matrix W')
parser.add_argument('--spu', type=float, default=0.6, help='sparsity of matrix U')
parser.add_argument('--sph', type=float, default=0.41, help='sparsity of hidden state h')
args = parser.parse_args()

# ngrp

for ngrp in range(1, int(args.ts/args.npe)):

    t_wx = args.xsize * args.hsize * (1-args.spw)
    t_uh = args.hsize * args.hsize * (1-args.spu) * (1-args.sph)
    t_prolog = t_wx * ngrp

    t_iter = t_prolog + max(0, t_uh * ngrp * (args.npe-1) - t_prolog * (ngrp-1)/ngrp)
    t_main = (math.ceil(args.ts/(args.npe-1)/ngrp)-1) * t_iter

    t_epilpg = ((args.ts-1)%((args.npe-1)*ngrp)+1)*t_uh

    clk = t_prolog + t_main + t_epilpg

    print(clk)