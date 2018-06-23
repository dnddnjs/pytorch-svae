from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import optim
from torch.utils.data import DataLoader

import os
import numpy as np
from collections import OrderedDict
from multiprocessing import cpu_count

from ptb import PTB
from model import SentenceVAE

parser = argparse.ArgumentParser(description='Sentence VAE Example')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--create_data', action='store_true')
parser.add_argument('--min_occ', type=int, default=1)

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--k', type=float, default=0.0025)
parser.add_argument('--x0', type=int, default=2500)
parser.add_argument('--log_interval', type=int, default=50)
parser.add_argument('--save_model_path', type=str, default='model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

splits = ['train', 'valid']
datasets = OrderedDict()

for split in splits:
    datasets[split] = PTB(
        data_dir=args.data_dir,
        split=split,
        create_data=args.create_data,
        max_sequence_length=60,
        min_occ=args.min_occ
    )

vocab_size = datasets['train'].vocab_size
sos_idx = datasets['train'].sos_idx
eos_idx = datasets['train'].eos_idx
pad_idx = datasets['train'].pad_idx

model = SentenceVAE(vocab_size, sos_idx, eos_idx, pad_idx, training=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def kl_anneal_function(step):
    k = args.k
    x0 = args.x0
    weight = float(1 / (1 + np.exp(-k * (step - x0))))
    return weight


NLL = torch.nn.NLLLoss(size_average=False,
                       ignore_index=datasets['train'].pad_idx)


def loss_function(reconx, x, mu, logvar, step):
    x = x.view(-1)
    reconx = reconx.view(-1, reconx.size(2))
    NLL_loss = NLL(reconx, x)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    beta = kl_anneal_function(step)
    loss = NLL_loss + beta * KLD
    return loss, NLL_loss, KLD, beta


def train(epoch, step):
    model.train()

    data_loader = DataLoader(
        dataset=datasets['train'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available()
    )

    train_loss = 0
    for batch_idx, batch in enumerate(data_loader):
        batch_size = batch['input'].size(0)
        if torch.cuda.is_available():
            batch['input'] = batch['input'].cuda()
            batch['target'] = batch['target'].cuda()
        logp, mu, logvar, z = model(batch['input'])
        loss, NLL_loss, KL_loss, KL_weight = loss_function(logp, batch['target'],
                                                           mu, logvar, step)

        loss = loss / batch_size
        if step == 10:
            checkpoint_path = os.path.join(args.save_model_path,
                                           "model_epoch_%i" % (epoch))
            torch.save(model.state_dict(), checkpoint_path)
            print("Model saved at %s" % checkpoint_path)
            

        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        step += 1

        if batch_idx % args.log_interval == 0:
            print('Train Epoch {} [{}/{}] Loss {:.2f} | NLL {:.2f}'
                  ' | KL {:.2f} | Beta {:.3f}'.format(epoch,
                   batch_idx * batch_size, len(data_loader.dataset),
                   loss.item(), NLL_loss.item() / batch_size,
                   KL_loss.item() / batch_size, KL_weight))

    print('====> Epoch: {} Average loss: {:.4f} steps: {}'.format(
          epoch, train_loss * args.batch_size / len(data_loader.dataset), step))

    checkpoint_path = os.path.join(args.save_model_path, "model_epoch_%i" % (epoch))
    torch.save(model.state_dict(), checkpoint_path)
    print("Model saved at %s" % checkpoint_path)
    return step


def test(step):
    model.eval()
    test_loss = 0

    data_loader = DataLoader(
        dataset=datasets['valid'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available()
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            if torch.cuda.is_available():
                batch['input'] = batch['input'].cuda()
                batch['target'] = batch['target'].cuda()
            logp, mu, logvar, z = model(batch['input'])
            loss, _, _, _ = loss_function(logp, batch['target'],
                                          mu, logvar, step)
            test_loss += loss.item()

    test_loss /= len(data_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


step = 0
for epoch in range(1, args.epochs + 1):
    step = train(epoch, step)
    test(step)
