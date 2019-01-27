import pickle as pk

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from nn_arch import cnn, rnn

from util import map_item


batch_size = 512

path_embed = 'feat/embed.pkl'
path_cnn_sent = 'feat/cnn_sent_train.pkl'
path_rnn_sent = 'feat/rnn_sent_train.pkl'
path_label = 'feat/label_train.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_cnn_sent, 'rb') as f:
    cnn_sents = pk.load(f)
with open(path_rnn_sent, 'rb') as f:
    rnn_sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)

funcs = {'cnn': cnn,
         'rnn': rnn}

paths = {'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5'}


def compile(name, embed_mat, seq_len):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                      weights=[embed_mat], input_length=seq_len, trainable=True)
    input = Input(shape=(seq_len,))
    embed_input = embed(input)
    func = map_item(name, funcs)
    output = func(embed_input, vocab_num)
    model = Model(input, output)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model


def fit(name, epoch, embed_mat, sents, labels):
    seq_len = len(sents[0])
    model = compile(name, embed_mat, seq_len)
    check_point = ModelCheckpoint(map_item(name, paths), monitor='val_loss', verbose=True, save_best_only=True)
    labels = np.expand_dims(labels, -1)
    model.fit(sents, labels, batch_size=batch_size, epochs=epoch,
              verbose=True, callbacks=[check_point], validation_split=0.2)


if __name__ == '__main__':
    fit('cnn', 50, embed_mat, cnn_sents, labels)
    fit('rnn', 50, embed_mat, rnn_sents, labels)
