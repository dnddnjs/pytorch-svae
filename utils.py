import numpy as np
from collections import Counter, OrderedDict


class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def idx2word(idx, i2w, pad_idx):

    sent_str = str()

    for i, word_id in enumerate(idx):
        if word_id == pad_idx:
            break
        sent_str += i2w[str(word_id)] + " "

    sent_str.strip()
    return sent_str


def interpolate(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s, e) in enumerate(zip(start, end)):
        interpolation[dim] = np.linspace(s, e, steps+2)

    return interpolation.T
