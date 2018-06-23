import json
import torch
import argparse
from ptb import PTB
from model import SentenceVAE
from utils import idx2word, interpolate
from collections import OrderedDict

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

model.load_state_dict(torch.load(args.load_checkpoint))
print("Model loaded from %s" % (args.load_checkpoint))

if torch.cuda.is_available():
    model = model.cuda()

model.eval()

print('----------SAMPLES----------')
for i in range(5):
    sample, z = model.inference()
    sample = sample.cpu().numpy()
    print(idx2word(sample, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

datasets = OrderedDict()
datasets['test'] = PTB(
    data_dir=args.data_dir,
    split='test',
    create_data=args.create_data,
    max_sequence_length=60,
    min_occ=args.min_occ
)

print('-------RECONSTRUCTION-------')

sample = datasets['test'].data['300']['input']
print('sample 1: ' + idx2word(sample[1:], i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
input = torch.Tensor(sample).long()
input = input.unsqueeze(0)
_, _, _, z = model(input)
recon, z = model.inference(z=z)
recon = recon.cpu().numpy()
print('reconst : ' + idx2word(recon, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

sample = datasets['test'].data['1500']['input']
print('sample 2: ' + idx2word(sample[1:], i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
input = torch.Tensor(sample).long()
input = input.unsqueeze(0)
_, _, _, z = model(input)
recon, z = model.inference(z=z)
recon = recon.cpu().numpy()
print('reconst : ' + idx2word(recon, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

'''
z1 = torch.randn([13]).numpy()
z2 = torch.randn([13]).numpy()
z = torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float()
samples, _ = model.inference(z=z)

print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
'''

