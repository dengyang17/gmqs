import pickle as pkl
import linecache
from random import Random

import numpy as np
import torch
import torch.utils.data as torch_data

import utils

num_samples = 1




class QADataset(torch_data.Dataset):

    def __init__(self, infos, config, indices=None):

        self.srcF = infos['srcF']
        self.tgtF = infos['tgtF']
        self.original_srcF = infos['original_srcF']
        self.original_tgtF = infos['original_tgtF']
        self.length = infos['length']
        self.ext_labelF = infos['ext_label']
        self.adjF = infos['adjF']
        self.infos = infos
        self.max_sent_len = config.max_sent_len
        self.max_sent_num = config.max_sent_num
        self.config = config
        if indices is None:
            self.indices = list(range(self.length))
        else:
            self.indices = indices

    def __getitem__(self, index):
        index = self.indices[index]

        src = [list(map(int, sent.split()))[:self.max_sent_len] for sent in eval(linecache.getline(self.srcF, index+1).strip())[:self.max_sent_num]]
        original_src = [sent.split()[:self.max_sent_len] for sent in eval(linecache.getline(self.original_srcF, index+1).strip())[:self.max_sent_num]]
        
        tgt = list(map(int, linecache.getline(self.tgtF, index+1).strip().split()))
        original_tgt = linecache.getline(self.original_tgtF, index+1).strip().split() 

        ext_label = list(map(int, eval(linecache.getline(self.ext_labelF, index+1).strip())[:self.max_sent_num-1])) # - 1 (question)

        adjs = []
        original_adjs = eval(linecache.getline(self.adjF, index+1).strip())
        for i in range(len(original_adjs)):
            if i in self.config.relations:
                adjs.append([list(map(float, row))[:self.max_sent_num] for row in original_adjs[i][:self.max_sent_num]])

        return src, tgt, original_src, original_tgt, ext_label, adjs

    def __len__(self):
        return len(self.indices)


def padding(data):
    src, tgt, original_src, original_tgt, ext_label, adjs = zip(*data)

    src_num = [len(s) for s in src]
    src_len = [[len(ss) for ss in s] for s in src]
    src_pad = torch.zeros(len(src), max(src_num), max([max(l) for l in src_len])).long()
    for i, s in enumerate(src):
        for j, ss in enumerate(s):
            end = src_len[i][j]
            src_pad[i, j, :end] = torch.LongTensor(ss)[:end]
    

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s)[:end]

    
    label_len = [len(s) for s in ext_label]
    ext_label_pad = torch.full((len(ext_label), max(label_len)), -1).long()
    for i, s in enumerate(ext_label):
        end = label_len[i]
        ext_label_pad[i, :end] = torch.LongTensor(s)[:end]
    

    adj_len = [len(adj[0]) for adj in adjs]
    adjs_pad = torch.zeros((len(adjs), len(adjs[0]), max(adj_len), max(adj_len))).float()
    for i, s in enumerate(adjs):
        for j, ss in enumerate(s):
            end = adj_len[i]
            adjs_pad[i, j, :end, :end] = torch.FloatTensor(ss)[:end, :end]

    return src_pad, tgt_pad, \
           original_src, original_tgt,\
           src_num, src_len, ext_label_pad, adjs_pad



