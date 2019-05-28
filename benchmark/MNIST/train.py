import os,sys 
sys.path.append(os.getcwd())
import logging
import torch 
import math 
import numpy as np 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
from model import RNN
import argparse
from utils.Param_transfer import set_to_zero_sparsity, set_to_zero_threshold, get_sparsity

parser = argparse.ArgumentParser(description='Train LSTM model for MNIST.')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit')
parser.add_argument('--nhid', type=int, default=128, help='number of hidden units per layer')
parser.add_argument('-ws', '--w_sp', type=float, nargs='+', default=[0, 0, 0, 0], help="Weight sparsity setting.")
parser.add_argument('-hs', '--h_sp', type=float, nargs='+', default=[0., 0.], help="Hidden state sparsity setting.")
parser.add_argument('-ht', '--h_th', type=float, nargs='+', default=[0., 0.], help="Hidden state threshold setting.")
parser.add_argument('-b', '--size_block', type=int, default=-1, help="Block size for hidden state sparsification.")
parser.add_argument('--retrain', action='store_true', help="Retrain or not.")
parser.add_argument('--model_path', default=None, help="The path to saved model")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# other parameters
sequence_length = 28
input_size = 28
num_layers = 2
num_classes = 10
batch_size = 100

torch.manual_seed(1111)

log_dir = './MNIST/logs/'
model_dir = './MNIST/models/'

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

###############################################################################
# Logging
###############################################################################
model_config_name = 'nhid:{}-nlayer:{}-epoch:{}'.format(
    args.nhid, num_layers, args.epochs
)

log_file_name = "{}.log".format(model_config_name)
log_file_path = os.path.join(log_dir, log_file_name)
logging.basicConfig(filename=log_file_path, 
                    level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    )
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))


###############################################################################
# Loading data
###############################################################################

train_dataset = torchvision.datasets.MNIST(root='./MNIST/data/', train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./MNIST/data/', train=False, download=True, transform=transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


model = RNN(input_size, args.nhid, num_layers, num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


###############################################################################
# Training
###############################################################################
def train(epochs, h_sp=[0.,0.], h_th=[0.,0.], block=-1):
    total_step = len(train_loader)
    for epoch in range(epochs):
        hidden = model.init_hidden(batch_size)
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            hidden = repackage_hidden(hidden)
            # Forward pass
            outputs, hidden, _ = model(images, hidden, h_sp=h_sp, h_th=h_th, block=block)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            if (i + 1) % 100 == 0:
                logger.info('|Epoch [{}/{}] | Step [{}/{}] | Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, total_step, loss.item()))

###############################################################################
# Test
###############################################################################
def evaluate(h_sp=[0.,0.], h_th=[0.,0.], block=-1):
    correct = 0
    total = 0

    hidden = (torch.zeros(num_layers, batch_size, args.nhid).to(device),
              torch.zeros(num_layers, batch_size, args.nhid).to(device))

    sparse_dict = {"LSTM1": 0., "LSTM2": 0.}
    iteration = 0

    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs, hidden, cur_dict = model(images, hidden, h_sp=h_sp, h_th=h_th, block=block)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        iteration += 1
        for k, v in sparse_dict.items():
            sparse_dict[k] += cur_dict[k]
        loss = criterion(outputs, labels)
    accuracy = 100.0 * correct / total
    print('|| Test Accuracy : {:.5f} || LSTM1 sparsity: {:.5f} || LSTM2 Sparsity: {:.5f} ||'.format(accuracy, sparse_dict['LSTM1']/iteration, sparse_dict['LSTM2']/iteration))
    return loss


# retrain flag==false
if args.retrain==False:
    print("Start a new training...")

    # trainning
    train(args.epochs)
    logger.info("Training process finish")

    # test
    with torch.no_grad():
        evaluate()

    # save model
    torch.save(model.state_dict(), './MNIST/models/{}.ckpt'.format(model_config_name))




###############################################################################
# Retraining code
###############################################################################

def sparsify_model(state_dict):
    state_mask_dict = {}
    for k, v in state_dict.items():
        if 'lstm1' in k:
            if 'weight_x' in k:
                state_dict[k] = set_to_zero_sparsity(v, sparsity=args.w_sp[0])
            else:
                state_dict[k] = set_to_zero_sparsity(v, sparsity=args.w_sp[1])
        elif 'lstm2' in k:
            if 'weight_x' in k:
                state_dict[k] = set_to_zero_sparsity(v, sparsity=args.w_sp[2])
            else:
                state_dict[k] = set_to_zero_sparsity(v, sparsity=args.w_sp[3])
        state_mask_dict[k] = (torch.abs(state_dict[k])>0)
    return state_mask_dict

def puring_model(state_dict, state_mask_dict):
    for k, v in state_dict.items():
        state_dict[k] = v * state_mask_dict[k].float()


# retrain flag==true
if args.retrain:
    print("Retraining...")

    best_val_loss = None

    if torch.cuda.is_available():
        state_dict = torch.load(args.model_path)
    else:
        state_dict = torch.load(args.model_path, map_location='cpu')
    # obtain the sparse mask
    state_mask_dict = sparsify_model(state_dict)
    retrained_model_path = args.model_path + '.retrain'

    try:
        for epoch in range(1, args.epochs+1):

            puring_model(state_dict, state_mask_dict)
            model.load_state_dict(state_dict)
            # retrain
            train(1, h_sp=args.h_sp, h_th=args.h_th, block=-1)
            val_loss = evaluate(h_sp=args.h_sp, h_th=args.h_th, block=args.size_block)

            if not best_val_loss or val_loss < best_val_loss:
                puring_model(state_dict, state_mask_dict)
                model.load_state_dict(state_dict)
                torch.save(model.state_dict(), retrained_model_path)
                best_val_loss = val_loss
            # else:
                # decrease the learning rate if needed

            # reload the model
            if torch.cuda.is_available():
                state_dict = torch.load(retrained_model_path)
            else:
                state_dict = torch.load(retrained_model_path, map_location='cpu')
    except KeyboardInterrupt:
        logger.error('Exiting from fine-tuning (retrain) early')

    # Run on test data.
    if torch.cuda.is_available():
        state_dict = torch.load(retrained_model_path)
    else:
        state_dict = torch.load(retrained_model_path, map_location='cpu')

    # test after puring
    puring_model(state_dict, state_mask_dict)
    model.load_state_dict(state_dict)
    test_loss = evaluate(h_sp=args.h_sp, h_th=args.h_th, block=args.size_block)