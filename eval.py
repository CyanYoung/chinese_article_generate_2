import pickle as pk

import numpy as np

import torch
import torch.nn.functional as F

from build import tensorize

from generate import models

from util import map_item


device = torch.device('cpu')

seq_len = 100

path_sent = 'feat/sent_test.pkl'
path_label = 'feat/label_test.pkl'
with open(path_sent, 'rb') as f:
    sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)


def test(name, sents, labels):
    sents, labels = tensorize([sents, labels], device)
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        probs = F.softmax(model(sents), dim=-1)
    probs = probs.numpy()
    len_sum, log_sum = [0] * 2
    for sent, label, prob in zip(sents, labels, probs):
        bound = sum(sent > 0).item()
        len_sum = len_sum + bound
        sent_log = 0
        for i in range(bound):
            sent_log = sent_log + np.log(prob[i][label[i]])
        log_sum = log_sum + sent_log
    print('\n%s %s %.2f' % (name, 'perp:', np.power(2, -log_sum / len_sum)))


if __name__ == '__main__':
    test('trm', sents, labels)
