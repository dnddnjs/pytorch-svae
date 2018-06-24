from nltk.tokenize import TweetTokenizer
from gensim.models import Word2Vec

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
                           min_count=1, workers=2, iter=1000, sg=1)

print(embedding_model.most_similar(positive=["negative"], topn=10))