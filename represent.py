import pickle as pk

import numpy as np

from gensim.models.word2vec import Word2Vec

from gensim.corpora import Dictionary

from util import flat_read


embed_len = 200
min_freq = 10
max_vocab = 5000
seq_len = 100

bos, eos = '<', '>'

pad_ind, oov_ind = 0, 1

path_word_vec = 'feat/word_vec.pkl'
path_word_ind = 'feat/word_ind.pkl'
path_embed = 'feat/embed.pkl'


def add_flag(texts):
    flag_texts = list()
    for text in texts:
        flag_texts.append(bos + text + eos)
    return flag_texts


def shift(flag_texts):
    sents = [text[:-1] for text in flag_texts]
    labels = [text[1:] for text in flag_texts]
    return sents, labels


def word2vec(texts, path_word_vec):
    model = Word2Vec(texts, size=embed_len, window=3, min_count=min_freq, negative=5, iter=10)
    word_vecs = model.wv
    with open(path_word_vec, 'wb') as f:
        pk.dump(word_vecs, f)
    if __name__ == '__main__':
        words = ['，', '。', '<', '>']
        for word in words:
            print(word_vecs.most_similar(word))


def tran_dict(word_inds, off):
    off_word_inds = dict()
    for word, ind in word_inds.items():
        off_word_inds[word] = ind + off
    return off_word_inds


def embed(sent_words, path_word_ind, path_word_vec, path_embed):
    model = Dictionary(sent_words)
    model.filter_extremes(no_below=min_freq, no_above=1.0, keep_n=max_vocab)
    word_inds = model.token2id
    word_inds = tran_dict(word_inds, off=2)
    with open(path_word_ind, 'wb') as f:
        pk.dump(word_inds, f)
    with open(path_word_vec, 'rb') as f:
        word_vecs = pk.load(f)
    vocab = word_vecs.vocab
    vocab_num = min(max_vocab + 2, len(word_inds) + 2)
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind] = word_vecs[word]
    with open(path_embed, 'wb') as f:
        pk.dump(embed_mat, f)


def sent2ind(words, word_inds, seq_len, keep_oov):
    seq = list()
    for word in words:
        if word in word_inds:
            seq.append(word_inds[word])
        elif keep_oov:
            seq.append(oov_ind)
    if len(seq) < seq_len:
        return seq + [pad_ind] * (seq_len - len(seq))
    else:
        return seq[-seq_len:]


def align(sent_words, path_sent):
    with open(path_word_ind, 'rb') as f:
        word_inds = pk.load(f)
    align_seqs = list()
    for words in sent_words:
        while len(words) > seq_len:
            trunc_words = words[:seq_len]
            trunc_seq = sent2ind(trunc_words, word_inds, seq_len, keep_oov=True)
            align_seqs.append(trunc_seq)
            words = words[seq_len:]
        pad_seq = sent2ind(words, word_inds, seq_len, keep_oov=True)
        align_seqs.append(pad_seq)
    align_seqs = np.array(align_seqs)
    with open(path_sent, 'wb') as f:
        pk.dump(align_seqs, f)


def vectorize(paths, mode, update):
    texts = flat_read(paths['data'], 'text')
    flag_texts = add_flag(texts)
    flag_text_words = [list(text) for text in flag_texts]
    if mode == 'train':
        if update:
            word2vec(flag_texts, path_word_vec)
        embed(flag_text_words, path_word_ind, path_word_vec, path_embed)
    sent_words, label_words = shift(flag_text_words)
    align(sent_words, paths['sent'])
    align(label_words, paths['label'])


if __name__ == '__main__':
    paths = dict()
    paths['data'] = 'data/train.csv'
    paths['sent'] = 'feat/sent_train.pkl'
    paths['label'] = 'feat/label_train.pkl'
    vectorize(paths, 'train', update=False)
    paths['data'] = 'data/dev.csv'
    paths['sent'] = 'feat/sent_dev.pkl'
    paths['label'] = 'feat/label_dev.pkl'
    vectorize(paths, 'dev', update=False)
    paths['data'] = 'data/test.csv'
    paths['sent'] = 'feat/sent_test.pkl'
    paths['label'] = 'feat/label_test.pkl'
    vectorize(paths, 'test', update=False)
