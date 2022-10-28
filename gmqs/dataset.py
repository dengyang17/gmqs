import os
import pickle

import torch

import utils


def load_data(config):
    """
    load data.
    update "data" due to the saved path in the pickle file
    :return: a dict with data and vocabulary
    """
    print("loading data...\n")
    data = pickle.load(open(config.data + "data.pkl", "rb"))
    # retrieve data, due to the problem of path.
    data["train"]["length"] = int(data["train"]["length"] * config.scale)
    data["train"]["srcF"] = os.path.join(config.data, "train.src.id")
    data["train"]["original_srcF"] = os.path.join(config.data, "train.src.str")
    data["train"]["tgtF"] = os.path.join(config.data, "train.tgt.id")
    data["train"]["original_tgtF"] = os.path.join(config.data, "train.tgt.str")
    data["test"]["srcF"] = os.path.join(config.data, "test.src.id")
    data["test"]["original_srcF"] = os.path.join(config.data, "test.src.str")
    data["test"]["tgtF"] = os.path.join(config.data, "test.tgt.id")
    data["test"]["original_tgtF"] = os.path.join(config.data, "test.tgt.str")
    # extractive label
    data["train"]["ext_label"] = os.path.join(config.data, "train.ext.label")
    data["test"]["ext_label"] = os.path.join(config.data, "test.ext.label")
    # graph data
    data["train"]["adjF"] = os.path.join(config.data, "train.adj")
    data["test"]["adjF"] = os.path.join(config.data, "test.adj")

    vocab = data["dict"]
    config.vocab_size = vocab.size()

    train_set = utils.QADataset(data["train"], config)
    valid_set = utils.QADataset(data["test"], config)


    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.padding,
    )
    if hasattr(config, "valid_batch_size"):
        valid_batch_size = config.valid_batch_size
    else:
        valid_batch_size = config.batch_size
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_set,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.padding,
    )
    return {
        "train_set": train_set,
        "valid_set": valid_set,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "vocab": vocab,
    }


def extend_vocab(vocab, original_src, original_tgt, src, tgt):
    oovs = None

    src_extend_ids = torch.zeros(src.size()).long()
    for i, s in enumerate(original_src):
        for j, ss in enumerate(s):
            sid, oovs = vocab.convertToIdxandOOVs(ss, utils.UNK_WORD, oovs=oovs)
            end = sid.size(0)
            src_extend_ids[i, j, :end] = sid
    
    tgt_extend_ids = torch.zeros(tgt.size()).long()
    for i, s in enumerate(original_tgt):
        sid, oovs = vocab.convertToIdxandOOVs(s, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD, oovs=oovs)
        end = sid.size(0)
        tgt_extend_ids[i, :end] = sid
    
    return src_extend_ids, tgt_extend_ids, oovs