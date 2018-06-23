from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


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
        self.word_dropout = nn.Dropout(p=1)

        self.encoder_rnn = nn.GRU(self.emb_size, self.h_size, batch_first=True)
        self.encode_fc1 = nn.Linear(self.h_size, self.z_size)
        self.encode_fc2 = nn.Linear(self.h_size, self.z_size)

        self.decode_fc1 = nn.Linear(self.z_size, self.h_size)
        self.decoder_rnn = nn.GRU(self.emb_size, self.h_size, batch_first=True)
        self.decode_fc2 = nn.Linear(self.h_size, self.vocab_size)

    def encode(self, x):
        x_emb = self.emb(x)
        _, h = self.encoder_rnn(x_emb)
        h = h.view(h.size(1), h.size(2))
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
        logp = logp.view(out.size(0), out.size(1), self.vocab_size)
        return logp

    def forward(self, x):
        mu, logvar, x_emb = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logp = self.decode(z, x_emb)
        return logp, mu, logvar, z

    def inference(self, z=None):
        if z is None:
            z = torch.randn([1, self.z_size])
            if torch.cuda.is_available():
                z = z.cuda()

        h = self.decode_fc1(z)
        h = h.unsqueeze(0)

        output = self.tensor(self.max_sequence_length).long()

        t = 0
        x = torch.Tensor(1).fill_(self.sos_idx).long()

        if torch.cuda.is_available():
            x = x.cuda()

        while t < self.max_sequence_length:
            x = x.unsqueeze(1)
            input_emb = self.emb(x)
            out, h = self.decoder_rnn(input_emb, h)
            logits = self.decode_fc2(out)

            _, x = torch.topk(logits, 1, dim=-1)
            x = x[0][0]
            if x[0] == 3:
                output = output[:t]
                break
            else:
                output[t] = x
            t += 1

        return output, z
