from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader

import os
import numpy as np
from collections import OrderedDict
from multiprocessing import cpu_count

from ptb import PTB
from utils import to_var

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
parser.add_argument('--save_model_path', type=str, default='bin')
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


class SentenceVAE(nn.Module):
    def __init__(self):
        super(SentenceVAE, self).__init__()
        if torch.cuda.is_available():
            self.tensor = torch.cuda.FloatTensor
        else:
            self.tensor = torch.Tensor

        self.max_sequence_length = 60
        self.vocab_size = datasets['train'].vocab_size
        self.sos_idx = datasets['train'].sos_idx
        self.eos_idx = datasets['train'].eos_idx
        self.pad_idx = datasets['train'].pad_idx

        self.z_size = 16
        self.h_size = 256
        self.emb_size = 300

        self.emb = nn.Embedding(self.vocab_size, self.emb_size)
        self.word_dropout = nn.Dropout(p=0.5)

        self.encoder_rnn = nn.GRU(self.emb_size, self.h_size, batch_first=True)
        self.encode_fc1 = nn.Linear(self.h_size, self.z_size)
        self.encode_fc2 = nn.Linear(self.h_size, self.z_size)

        self.decode_fc1 = nn.Linear(self.z_size, self.h_size)
        self.decoder_rnn = nn.GRU(self.emb_size, self.h_size, batch_first=True)
        self.decode_fc2 = nn.Linear(self.h_size, self.vocab_size)

    def encode(self, x, len):
        sorted_len, sorted_idx = torch.sort(len, descending=True)
        x = x[sorted_idx]

        x_emb = self.emb(x)
        x = rnn_utils.pack_padded_sequence(x_emb, sorted_len.data.tolist(),
                                           batch_first=True)
        _, h = self.encoder_rnn(x)
        h = h.squeeze()
        mu = self.encode_fc1(h)
        logvar = self.encode_fc2(h)
        return mu, logvar, x_emb, sorted_len, sorted_idx

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, input_emb, sorted_len, sorted_idx):
        h = self.decode_fc1(z)
        h = h.unsqueeze(0)

        input_emb = self.word_dropout(input_emb)
        y = rnn_utils.pack_padded_sequence(input_emb, sorted_len.data.tolist(),
                                           batch_first=True)
        out, _ = self.decoder_rnn(y, h)

        out = rnn_utils.pad_packed_sequence(out, batch_first=True)[0]
        out = out.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        out = out[reversed_idx]

        logp = F.log_softmax(self.decode_fc2(out.view(-1, out.size(2))), dim=-1)
        logp = logp.view(out.size(0), out.size(1), self.emb.num_embeddings)
        return logp

    def forward(self, x, len):
        mu, logvar, input_emb, sorted_len, sorted_idx = self.encode(x, len)
        z = self.reparameterize(mu, logvar)
        logp = self.decode(z, input_emb, sorted_len, sorted_idx)
        return logp, mu, logvar, z


model = SentenceVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def kl_anneal_function(step):
    k = args.k
    x0 = args.x0
    weight = float(1 / (1 + np.exp(-k * (step - x0))))
    return weight


NLL = torch.nn.NLLLoss(size_average=False,
                       ignore_index=datasets['train'].pad_idx)


def loss_function(reconx, x, length, mu, logvar, step):
    # cut-off unnecessary padding from target, and flatten
    x = x[:, :torch.max(length).data].contiguous().view(-1)
    reconx = reconx.view(-1, reconx.size(2))

    # Negative Log Likelihood
    NLL_loss = NLL(reconx, x)

    # KL Divergence
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

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = to_var(v)

        # Forward pass
        logp, mu, logvar, z = model(batch['input'], batch['length'])

        # loss calculation
        loss, NLL_loss, KL_loss, KL_weight = loss_function(logp, batch['target'],
                                                           batch['length'], mu,
                                                           logvar, step)

        loss = loss / batch_size

        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        step += 1

        if batch_idx % args.log_interval == 0:
            print(
                "Train Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                % (batch_idx, len(data_loader) - 1,
                   loss.data[0], NLL_loss.data[0] / batch_size,
                   KL_loss.data[0] / batch_size, KL_weight))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss * args.batch_size / len(data_loader.dataset)))

    checkpoint_path = os.path.join(args.save_model_path, "E%i.pytorch" % (epoch))
    torch.save(model.state_dict(), checkpoint_path)
    print("Model saved at %s" % checkpoint_path)
    return step


def test(epoch, step):
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
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)

            # Forward pass
            logp, mu, logvar, z = model(batch['input'], batch['length'])

            # loss calculation
            loss, NLL_loss, KL_loss, KL_weight = loss_function(logp,
                                                               batch['target'],
                                                               batch['length'],
                                                               mu,
                                                               logvar, step)

            test_loss += loss.item()

    test_loss /= len(data_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


step = 0
for epoch in range(1, args.epochs + 1):
    step = train(epoch, step)
    test(epoch, step)
