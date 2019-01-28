import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Trm(nn.Module):
    def __init__(self, embed_mat, pos_mat, mask_mat, head, stack):
        super(Trm, self).__init__()
        self.decode = TrmDecode(embed_mat, pos_mat, mask_mat, head, stack)

    def forward(self, y):
        return self.decode(y)


class TrmDecode(nn.Module):
    def __init__(self, embed_mat, pos_mat, mask_mat, head, stack):
        super(TrmDecode, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.pos, self.mask = pos_mat, mask_mat
        self.layers = nn.ModuleList([DecodeLayer(embed_len, head) for _ in range(stack)])
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, vocab_num))

    def forward(self, y):
        p, m = self.pos.repeat(y.size(0), 1, 1), self.mask.repeat(y.size(0), 1, 1, 1)
        y = self.embed(y)
        y = y + p
        for layer in self.layers:
            y = layer(y, m)
        return self.dl(y)


class DecodeLayer(nn.Module):
    def __init__(self, embed_len, head):
        super(DecodeLayer, self).__init__()
        self.head = head
        self.qry = nn.Linear(embed_len, 200 * head)
        self.key = nn.Linear(embed_len, 200 * head)
        self.val = nn.Linear(embed_len, 200 * head)
        self.fuse = nn.Linear(200 * head, 200)
        self.lal = nn.Sequential(nn.Linear(200, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 200))
        self.lns = nn.ModuleList([nn.LayerNorm(200) for _ in range(2)])

    def mul_att(self, x, y, m):
        q = self.qry(y).view(y.size(0), y.size(1), self.head, -1).transpose(1, 2)
        k = self.key(x).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        v = self.val(x).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        d = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        d = d.masked_fill(m, -float('inf'))
        a = F.softmax(d, dim=-1)
        c = torch.matmul(a, v).transpose(1, 2)
        c = c.contiguous().view(c.size(0), c.size(1), -1)
        return self.fuse(c)

    def forward(self, y, m):
        r = y
        y = self.mul_att(y, y, m)
        y = self.lns[0](y + r)
        r = y
        y = self.lal(y)
        return self.lns[1](y + r)
