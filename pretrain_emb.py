from nltk.tokenize import TweetTokenizer
from gensim.models import Word2Vec, KeyedVectors
import torch
from torch import nn

tokenizer = TweetTokenizer(preserve_case=False)
tokenized_contents = []
with open('data/ptb.train.txt', 'r') as file:
    for i, line in enumerate(file):
        words = tokenizer.tokenize(line)
        tokenized_contents.append(words)
print('complete tokenize train data')

with open('data/ptb.valid.txt', 'r') as file:
    for i, line in enumerate(file):
        words = tokenizer.tokenize(line)
        tokenized_contents.append(words)
print('complete tokenize validation data')

with open('data/ptb.test.txt', 'r') as file:
    for i, line in enumerate(file):
        words = tokenizer.tokenize(line)
        tokenized_contents.append(words)
print('complete tokenize test data')


embedding_model = Word2Vec(tokenized_contents, size=300, window=2,
                           min_count=1, workers=8, iter=10, sg=1)

embedding_model.wv.save('model/pretrained_embedding')
print('train over and saved model')

embedding = KeyedVectors.load('model/pretrained_embedding')
weights = torch.FloatTensor(embedding.syn0)
embedding = nn.Embedding.from_pretrained(weights)
input = torch.LongTensor([1])
embedding(input)