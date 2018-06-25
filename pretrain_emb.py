from gensim.models import Word2Vec
import gensim
import glob
import os

tokenized_contents = []
for filename in glob.glob(os.path.join('data/', '*.txt')):
    with open(filename, 'rb') as file:
        print(filename)
        for i, line in enumerate(file):
            words = gensim.utils.simple_preprocess(line)
            tokenized_contents.append(words)

print('complete tokenize text data')
tokenized_contents += ['<pad>', '<sos>', '<eos>']

embedding_model = Word2Vec(tokenized_contents, size=300, window=2,
                           min_count=1, workers=8, iter=10, sg=1)

embedding_model.wv.save('model/pretrained_embedding')
print('train over and saved model')