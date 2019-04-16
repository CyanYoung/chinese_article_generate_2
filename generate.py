import pickle as pk

import numpy as np
from numpy.random import choice

import torch
import torch.nn.functional as F

from represent import sent2ind

from util import map_item


def ind2word(word_inds):
    ind_words = dict()
    for word, ind in word_inds.items():
        ind_words[ind] = word
    return ind_words


device = torch.device('cpu')

seq_len = 100
min_len = 20
max_len = 100

bos, eos = '<', '>'

path_word_ind = 'feat/word_ind.pkl'
with open(path_word_ind, 'rb') as f:
    word_inds = pk.load(f)

eos_ind = word_inds[eos]

puncs = ['，', '。']
punc_inds = [word_inds[punc] for punc in puncs]

ind_words = ind2word(word_inds)

paths = {'trm': 'model/trm.pkl'}

models = {'trm': torch.load(map_item('trm', paths), map_location=device)}


def sample(probs, count, cand):
    max_probs = np.array(sorted(probs, reverse=True)[:cand])
    max_probs = max_probs / np.sum(max_probs)
    max_inds = np.argsort(-probs)[:cand]
    if max_inds[0] in punc_inds:
        next_ind = max_inds[0]
    elif count < min_len:
        next_ind = eos_ind
        while next_ind == eos_ind:
            next_ind = choice(max_inds, p=max_probs)
    else:
        next_ind = choice(max_inds, p=max_probs)
    return ind_words[next_ind]


def predict(text, name):
    text = bos + text
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        next_word, count = '', len(text) - 1
        while next_word != eos and count < max_len:
            text = text + next_word
            count = count + 1
            pad_seq = sent2ind(text, word_inds, seq_len, keep_oov=True)
            sent = torch.LongTensor([pad_seq]).to(device)
            step = min(count - 1, seq_len - 1)
            prods = model(sent)[0][step]
            probs = F.softmax(prods, dim=0).numpy()
            next_word = sample(probs, count, cand=5)
    return text[1:]


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('trm: %s' % predict(text, 'trm'))
