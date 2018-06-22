import json
import torch
import argparse

from model import SentenceVAE
from utils import to_var, idx2word, interpolate

parser = argparse.ArgumentParser(description='Sentence VAE Example')

parser.add_argument('-c', '--load_checkpoint', type=str)
parser.add_argument('-n', '--num_samples', type=int, default=10)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--create_data', action='store_true')
parser.add_argument('--min_occ', type=int, default=1)

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

with open(args.data_dir + '/ptb.vocab.json', 'r') as file:
    vocab = json.load(file)

w2i, i2w = vocab['w2i'], vocab['i2w']

model = SentenceVAE(
    vocab_size=len(w2i),
    sos_idx=w2i['<sos>'],
    eos_idx=w2i['<eos>'],
    pad_idx=w2i['<pad>']
)

model.load_state_dict(torch.load("bin/E4.pytorch"))
print("Model loaded from %s" % ("bin/E4.pytorch"))

if torch.cuda.is_available():
    model = model.cuda()

model.eval()

samples, z = model.inference(n=args.num_samples)
print('----------SAMPLES----------')
print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

z1 = torch.randn([16]).numpy()
z2 = torch.randn([16]).numpy()
z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
samples, _ = model.inference(z=z)
print('-------INTERPOLATION-------')
print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')