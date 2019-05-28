import argparse
import time
import math
import os, sys
sys.path.append(os.getcwd())

import logging
import torch
import torch.nn as nn
import torch.onnx

import data
import model

from utils.Param_transfer import set_to_zero_sparsity, set_to_zero_threshold, get_sparsity

parser = argparse.ArgumentParser(description='PyTorch PTB RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./LM/data/PTB/', help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=35, help='sequence length')
parser.add_argument('--dropout', type=float, default=0.65, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')

parser.add_argument('-ws', '--w_sp', type=float, nargs='+', default=[0, 0, 0, 0], help="Weight sparsity setting.")
parser.add_argument('-hs', '--h_sp', type=float, nargs='+', default=[0., 0.], help="Hidden state sparsity setting.")
parser.add_argument('-ht', '--h_th', type=float, nargs='+', default=[0., 0.], help="Hidden state threshold setting.")
parser.add_argument('-b', '--size_block', type=int, default=-1, help="Block size for hidden state sparsification.")

parser.add_argument('--retrain', action='store_true', help="Retrain or not.")
parser.add_argument('--model_path', default=None, help="The path to saved model")

parser.add_argument('--l1', action='store_true', help="Whether to add l1 regularization")
parser.add_argument('--l1_lambda', type=float, default=0.00001, help="lambda of l1 regularization loss")

parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--save_dir', type=str, default='./LM/models/PTB/', help='path to save the final model')
parser.add_argument('--log_dir', type=str, default='./LM/logs/PTB/', help='path for log')
parser.add_argument('--onnx-export', type=str, default='', help='path to export the final model in onnx format')
args = parser.parse_args()


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Logging
###############################################################################
model_config_name = "model:{}-em:{}-nhid:{}-nlayers:{}-bptt:{}-epoch:{}-lr:{}-tied:{}-l1:{}-l1_lambda:{}-dropout:{}".format(
    args.model, args.emsize, args.nhid, args.nlayers, 
    args.bptt, args.epochs, args.lr, args.tied, args.l1, args.l1_lambda, args.dropout)

log_file_name = "{}.log".format(model_config_name)
log_file_path = os.path.join(args.log_dir, log_file_name)
logging.basicConfig(filename=log_file_path, 
                    level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    )
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.info(args)

###############################################################################
# Model saving
###############################################################################
saver_filename = "{}.ckpt".format(model_config_name)
saver_file_path = os.path.join(args.save_dir, saver_filename)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source, h_sp=[0.,0.], h_th=[0.,0.], block=-1):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden, sparse=True, h_sp=h_sp, h_th=h_th, block=block)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(optimizer=None, h_sp=[0.,0.], h_th=[0.,0.], block=-1, lr=args.lr):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        if not optimizer:
            model.zero_grad()
        else:
            optimizer.zero_grad()

        output, hidden = model(data, hidden, sparse=True, h_sp=h_sp, h_th=h_th, block=block)
        loss = criterion(output.view(-1, ntokens), targets)

        # Add l1 regularization
        if args.l1:
            l1_regularization = torch.tensor(0, dtype=torch.float32).to(device)
            lambda_l1 = torch.tensor(args.l1_lambda).to(device)
            for param in model.parameters():
                l1_regularization += torch.norm(param, 1).to(device)
            loss = loss + lambda_l1 * l1_regularization

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if not optimizer:
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)
        else: 
            optimizer.step()
            lr = get_lr(optimizer)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:5.2e} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):
    logger.info('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# retrain flag==false
if args.retrain==False:
    print("Start a new training...")
    # Loop over epochs.
    lr = args.lr
    best_val_loss = None

    # FIXME: we do not use the built-in optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=1)

    # load the model and continue training
    if args.model_path:

        # synchronize the model path
        saver_file_path = args.model_path
        if torch.cuda.is_available():
            state_dict = torch.load(args.model_path)
        else:
            state_dict = torch.load(args.model_path, map_location='cpu')
        # load the parameters to model
        model.load_state_dict(state_dict)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(lr=lr)
            val_loss = evaluate(data_source=val_data)
            # scheduler.step(val_loss)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | ' 'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
            logger.info('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                torch.save(model.state_dict(), saver_file_path)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        logger.error('-' * 89)
        logger.error('Exiting from training early')


    ###############################################################################
    # Testing
    ###############################################################################
    # load the best evaluated model
    with open(saver_file_path, 'rb') as f:
        if torch.cuda.is_available():
            state_dict = torch.load(saver_file_path)
        else:
            state_dict = torch.load(saver_file_path, map_location = 'cpu')
        model.load_state_dict(state_dict)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(data_source=test_data)
    logger.info('=' * 89)
    logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logger.info('=' * 89)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args. bptt)



###############################################################################
# Retraining code
###############################################################################

def sparsify_model(state_dict):
    state_mask_dict = {}
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
        state_mask_dict[k] = (torch.abs(state_dict[k])>0)
    return state_mask_dict

def puring_model(state_dict, state_mask_dict):
    for k, v in state_dict.items():
        state_dict[k] = v * state_mask_dict[k].float()


# retrain flag==true
if args.retrain:
    print("Retraining...")
    # number of epoch is equal to the original train epoch/4
    fine_tune_epoch = int(args.epochs/3)

    lr = args.lr
    best_val_loss = None

    if torch.cuda.is_available():
        state_dict = torch.load(args.model_path)
    else:
        state_dict = torch.load(args.model_path, map_location='cpu')
    # obtain the sparse mask
    state_mask_dict = sparsify_model(state_dict)

    retrained_model_path = args.model_path + '.retrain'

    try:
        for epoch in range(1, fine_tune_epoch+1):
            epoch_start_time = time.time()

            puring_model(state_dict, state_mask_dict)
            model.load_state_dict(state_dict)
            # retrain
            train(h_sp=args.h_sp, h_th=args.h_th, block=-1, lr=lr)
            val_loss = evaluate(data_source=val_data, h_sp=args.h_sp, h_th=args.h_th, block=-1)
            # scheduler.step(val_loss)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
            logger.info('-' * 89)
            if not best_val_loss or val_loss < best_val_loss:
                torch.save(model.state_dict(), retrained_model_path)
                best_val_loss = val_loss
            else:
                lr /= 4.0
            # reload the model
            if torch.cuda.is_available():
                state_dict = torch.load(retrained_model_path)
            else:
                state_dict = torch.load(retrained_model_path, map_location='cpu')

    except KeyboardInterrupt:
        logger.error('-' * 89)
        logger.error('Exiting from fine-tuning (retrain) early')

    # Run on test data.
    if torch.cuda.is_available():
        state_dict = torch.load(retrained_model_path)
    else:
        state_dict = torch.load(retrained_model_path, map_location='cpu')

    # test after puring
    puring_model(state_dict, state_mask_dict)
    model.load_state_dict(state_dict)
    test_loss = evaluate(data_source=test_data, h_sp=args.h_sp, h_th=args.h_th, block=args.size_block)
    logger.info('=' * 89)
    logger.info('| End of fine-tuning | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logger.info('=' * 89)