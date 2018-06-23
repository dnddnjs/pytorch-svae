from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils

from utils import to_var


class SentenceVAE(nn.Module):
    def __init__(self, vocab_size, sos_idx, eos_idx, pad_idx):
        super(SentenceVAE, self).__init__()
        if torch.cuda.is_available():
            self.tensor = torch.cuda.FloatTensor
        else:
            self.tensor = torch.Tensor

        self.max_sequence_length = 60

        self.vocab_size = vocab_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        self.z_size = 13
        self.h_size = 191
        self.emb_size = 353

        self.emb = nn.Embedding(self.vocab_size, self.emb_size)
        self.word_dropout = nn.Dropout(p=0.62)

        self.encoder_rnn = nn.GRU(self.emb_size, self.h_size, batch_first=True)
        self.encode_fc1 = nn.Linear(self.h_size, self.z_size)
        self.encode_fc2 = nn.Linear(self.h_size, self.z_size)

        self.decode_fc1 = nn.Linear(self.z_size, self.h_size)
        self.decoder_rnn = nn.GRU(self.emb_size, self.h_size, batch_first=True)
        self.decode_fc2 = nn.Linear(self.h_size, self.vocab_size)

    def encode(self, x):
        x_emb = self.emb(x)
        _, h = self.encoder_rnn(x_emb)
        h = h.squeeze()
        mu = self.encode_fc1(h)
        logvar = self.encode_fc2(h)
        return mu, logvar, x_emb

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, x_emb):
        h = self.decode_fc1(z)
        h = h.unsqueeze(0)

        x_emb = self.word_dropout(x_emb)
        out, _ = self.decoder_rnn(x_emb, h)
        out = out.contiguous()
        logit = self.decode_fc2(out.view(-1, out.size(2)))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(out.size(0), out.size(1), self.emb.num_embeddings)
        return logp

    def forward(self, x):
        mu, logvar, x_emb = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logp = self.decode(z, x_emb)
        return logp, mu, logvar, z

    def inference(self, n=4, z=None):
        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.z_size]))
        else:
            batch_size = z.size(0)

        h = self.decode_fc1(z)
        h = h.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:

            if t == 0:
                x = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            x = x.unsqueeze(1)
            input_emb = self.emb(x)
            out, h = self.decoder_rnn(input_emb, h)

            logits = self.decode_fc2(out)

            _, x = torch.topk(logits, 1, dim=-1)
            x = x.squeeze()

            # save next input
            running_latest = generations[sequence_running]
            # update token at position t
            running_latest[:, t] = x.data
            # save back
            generations[sequence_running] = running_latest

            # update gloabl running sequence
            sequence_mask[sequence_running] = (x != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (x != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            # print(running_mask)
            if len(running_seqs) > 0:
                try:
                    x.size(0)
                except RuntimeError:
                    x = x.unsqueeze(0)

                x = x[running_seqs]
                h = h[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs),
                                            out=self.tensor()).long()

            t += 1

        return generations, z
