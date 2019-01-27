import pickle as pk

import numpy as np
from numpy.random import choice

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from util import map_item


def ind2word(word_inds):
    ind_words = dict()
    for word, ind in word_inds.items():
        ind_words[ind] = word
    return ind_words


win_len = 10
seq_len = 100
min_len = 20
max_len = 100

bos, eos = '<', '>'

path_word_ind = 'feat/word2ind.pkl'
with open(path_word_ind, 'rb') as f:
    word_inds = pk.load(f)

eos_ind = word_inds[eos]

puncs = ['，', '。']
punc_inds = [word_inds[punc] for punc in puncs]

ind_words = ind2word(word_inds)

paths = {'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5'}

models = {'cnn': load_model(map_item('cnn', paths)),
          'rnn': load_model(map_item('rnn', paths))}


def sample(probs, sent_len, cand):
    max_probs = np.array(sorted(probs, reverse=True)[:cand])
    max_probs = max_probs / np.sum(max_probs)
    max_inds = np.argsort(-probs)[:cand]
    if max_inds[0] in punc_inds:
        next_ind = max_inds[0]
    elif sent_len < min_len:
        next_ind = eos_ind
        while next_ind == eos_ind:
            next_ind = choice(max_inds, p=max_probs)
    else:
        next_ind = choice(max_inds, p=max_probs)
    return ind_words[next_ind]


def predict(text, name):
    sent = bos + text.strip()
    model = map_item(name, models)
    pad_len = seq_len + win_len - 1 if name == 'cnn' else seq_len
    next_word = ''
    while next_word != eos and len(sent) < max_len:
        sent = sent + next_word
        seq = word2ind.texts_to_sequences([sent])[0]
        align_seq = pad_sequences([seq], maxlen=pad_len)
        probs = model.predict(align_seq)[0][-1]
        next_word = sample(probs, len(sent), cand=5)
    return sent[1:]


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('cnn: %s' % predict(text, 'cnn'))
        print('rnn: %s' % predict(text, 'rnn'))
