import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models
import utils
from models import rnn


MAX_SIZE = 200


class PositionalEncoding(nn.Module):
    """ positional encoding """

    def __init__(self, dropout, dim, max_len=1000):
        """
        initialization of required variables and functions
        :param dropout: dropout probability
        :param dim: hidden size
        :param max_len: maximum length
        """
        # positional encoding initialization
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        # term to divide
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        # sinusoidal positional encoding
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """
        create positional encoding
        :param emb: word embedding
        :param step: step for decoding in inference
        :return: positional encoding representation
        """
        # division of size
        emb = emb * math.sqrt(self.dim)
        if step is None:
            # residual connection
            emb = emb + self.pe[:,:emb.size(1)]   # [batch, len, size]
        else:
            # step for inference
            emb = emb + self.pe[:,step]   # [batch, len, size]
        emb = self.dropout(emb)
        return emb


class PositionwiseFeedForward(nn.Module):
    """ Point-wise Feed-Forward NN, FFN, in fact 1-d convolution """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        initialization of required functions
        :param d_model: model size
        :param d_ff: intermediate size
        :param dropout: dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        run FFN
        :param x: input
        :return: output
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        # with residual connection
        return output + x


class TransformerEncoderLayer(nn.Module):
    """ Transformer encoder layer """

    def __init__(self, config):
        """
        initialization of required variables and functions
        :param config: configuration
        """
        super(TransformerEncoderLayer, self).__init__()
        self.config = config
        # self attention
        self.self_attn = models.Multihead_Attention(
            model_dim=config.hidden_size, head_count=config.heads, dropout=config.dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=config.hidden_size, d_ff=config.d_ff, dropout=config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs, mask):
        """
        run transformer encoder layer
        :param inputs: inputs
        :param mask: mask
        :return: output
        """
        # self attention
        input_norm = self.layer_norm(inputs)  # [batch, len, size]
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                        mask=mask)  # [batch, len, size]
        out = self.dropout(context) + inputs    # [batch, len, size]
        # FFN
        return self.feed_forward(out)   # [batch, len, size]


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask=None):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        if mask is not None:
            att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1) # bsz, input_len, memory_len
        output_one = torch.bmm(weight_one, memory) # bsz, memory_len, hidden_size
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len) # bsz, 1, input_len
        output_two = torch.bmm(weight_two, input) # bsz, 1, hidden_size

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)


class TransformerEncoder(nn.Module):
    """ Transformer encoder """

    def __init__(self, config, padding_idx=0):
        """
        initialization of required variables and functions
        :param config: configuration
        :param padding_idx: index for padding in the dictionary
        """
        super(TransformerEncoder, self).__init__()

        self.config = config
        self.num_layers = config.enc_num_layers

        # HACK: 512 for word embeddings, 512 for condition embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.emb_size,
                                      padding_idx=padding_idx)
        # positional encoding
        self.position_embedding = PositionalEncoding(
                config.dropout, config.emb_size)

        # transformer
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(config)
             for _ in range(config.enc_num_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.padding_idx = padding_idx

    def forward(self, src):
        """
        run transformer encoder
        :param src: source input
        :return: output
        """
        embed = self.embedding(src)

        out = self.position_embedding(embed)    # [batch, len, size]
 
        src_words = src  # [batch, len]
        src_batch, src_len = src_words.size()
        padding_idx = self.padding_idx
        mask = src_words.data.eq(padding_idx).unsqueeze(1).expand(src_batch, src_len, src_len)    # [batch, len, len]

        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)  # [batch, len, size]

        return out


# Decoder
class TransformerDecoderLayer(nn.Module):
    """ Transformer decoder layer """

    def __init__(self, config):
        """
        initialization for required variables and functions
        :param config: configuration
        """
        super(TransformerDecoderLayer, self).__init__()
        self.config = config
        # self attention
        self.self_attn = models.Multihead_Attention(
            model_dim=config.hidden_size, head_count=config.heads, dropout=config.dropout)

        self.context_attn = models.Multihead_Attention(
            model_dim=config.hidden_size, head_count=config.heads, dropout=config.dropout)
        
        self.document_attn = models.Multihead_Attention(
            model_dim=config.hidden_size, head_count=config.heads, dropout=config.dropout)
        
        self.feed_forward = PositionwiseFeedForward(
            config.hidden_size, config.d_ff, config.dropout)
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        self.dropout = config.dropout
        self.drop = nn.Dropout(config.dropout)
        self.drop_d = nn.Dropout(config.dropout)

        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)


    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, doc_memory_bank, doc_pad_mask,
                layer_cache=None, step=None):
        """
        run transformer decoder layer
        :param inputs: inputs
        :param memory_bank: source representations
        :param src_pad_mask: source padding mask
        :param tgt_pad_mask: target padding mask
        :param layer_cache: layer cache for decoding in inference stage
        :param step: step for decoding in inference stage
        :return: output, attention weights and input norm
        """
        dec_mask = torch.gt(tgt_pad_mask
                            + self.mask[:, :tgt_pad_mask.size(1), :tgt_pad_mask.size(1)], 0)

        # self attention
        input_norm = self.layer_norm_1(inputs)
        query, _ = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         type="self")   # [batch, q_len, size]
        # residual connection
        query = self.drop(query) + inputs   # [batch, q_len, size]

        # context attention
        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                          mask=src_pad_mask,
                                          layer_cache=layer_cache,
                                          type="context",
                                          Bernoulli=self.config.Bernoulli)  # [batch, q_len, size]
        mid = self.drop(mid) + query

        # document attention
        mid_norm = self.layer_norm_3(mid)
        doc_mid, doc_attn = self.document_attn(doc_memory_bank, doc_memory_bank, mid_norm,
                                          mask=doc_pad_mask,
                                          layer_cache=layer_cache,
                                          type="document",
                                          Bernoulli=self.config.Bernoulli)  # [batch, q_len, size]
        mid = self.drop_d(doc_mid) + mid
        
        output = self.feed_forward(mid)  # [batch, q_len, size]
        return output, attn

    def _get_attn_subsequent_mask(self, size):
        """
        get mask for target
        :param size: max size
        :return: target mask
        """
        attn_shape = (1, size, size)    # [1, size, size]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    """ Transformer decoder """

    def __init__(self, config, tgt_embedding=None, padding_idx=0):
        """
        initialization for required variables and functions
        :param config: configuration
        :param tgt_embedding: target embedding
        :param padding_idx: padding index
        """
        super(TransformerDecoder, self).__init__()
        self.config = config
        self.num_layers = config.dec_num_layers
        if tgt_embedding:
            self.embedding = tgt_embedding
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.emb_size,
                                          padding_idx=padding_idx)
        if config.positional:
            self.position_embedding = PositionalEncoding(
                config.dropout, config.emb_size)
        else:
            self.rnn = nn.LSTMCell(config.emb_size, config.hidden_size)

        self.padding_idx = padding_idx
        # state to store elements, including source and layer cache
        self.state = {}
        # transformer decoder
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(config)
             for _ in range(config.dec_num_layers)])
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        if config.pointer:
            pointer_num = 3
            self.question_attention = models.External_Attention(config.hidden_size)
            self.document_attention = models.Hierarchical_Attention(config.hidden_size)
            self.linear = nn.Linear(config.hidden_size * pointer_num, pointer_num)
            
            
    def forward(self, tgt, memory_bank, doc_memory_bank, sent_vec, sent_pad_mask, state=None, step=None):
        """
        run transformer decoder
        :param tgt: target input
        :param memory_bank: source representations
        :param state: state
        :param step: step for inference
        :return: output, attention weights and state
        """
        src = self.state["src"]
        src_words = src  # [batch, src_len]
        tgt_words = tgt  # [batch, tgt_len]
        sent_batch, sent_len = sent_pad_mask.size()
        src_words = src_words.view(sent_batch, sent_len+1, -1)
        doc_words = src_words[:, 1:].view(sent_batch, -1)
        src_words = src_words[:, 0].view(sent_batch, -1)

        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()
        doc_batch, doc_len = doc_words.size()

        emb = self.embedding(tgt)   # [batch, tgt_len, size]
        emb = self.position_embedding(emb, step=step)

        output = emb   # [batch, tgt_len, size]
        src_memory_bank = memory_bank   # [batch, src_len, size]

        padding_idx = self.padding_idx
        # source padding mask
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, src_len)    # [batch, tgt_len, src_len]
        # target padding mask
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)    # [batch, tgt_len, tgt_len]
        # document padding mask
        doc_pad_mask = doc_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(doc_batch, tgt_len, doc_len)    # [batch, tgt_len, src_len]
        # sentence padding mask
        sent_pad_mask = sent_pad_mask.unsqueeze(1).expand(sent_batch, tgt_len, sent_len)        
        
        # run transformer decoder layers
        for i in range(self.num_layers):
            output, _ = self.transformer_layers[i](
                output, src_memory_bank,
                src_pad_mask, tgt_pad_mask,
                doc_memory_bank, doc_pad_mask,
                layer_cache=self.state["cache"]["layer_{}".format(i)],
                step=step)
        
        output = self.layer_norm(output)    # [batch, tgt_len, size]

        if self.config.pointer:
            question_context, attn = self.question_attention(src_memory_bank, src_memory_bank, output, src_pad_mask)
            document_context, doc_attn = self.document_attention(doc_memory_bank, sent_vec, output, doc_pad_mask, sent_pad_mask)
            pointers = F.softmax(self.linear(torch.cat([output, question_context, document_context], dim=-1)), dim=-1)

            return output, attn, doc_attn, pointers
        else:
            return output, attn, doc_attn, None
            # [batch, tgt_len, size], [batch, tgt_len, src_len]
    
    def map_state(self, fn):
        """
        state mapping
        :param fn: function
        :return: none
        """
        def _recursive_map(struct, batch_dim=0):
            """
            recursive mapping
            :param struct: object for mapping
            :param batch_dim: batch dimension
            :return: none
            """
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)
        # self.state["src"] = fn(self.state["src"], 1)
        # layer cache mapping
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])


def init_state(self, src, memory_bank, num_layers):
    """
    state initialization, to replace the one in the transformer decoder
    :param self: self
    :param src: source input
    :param memory_bank: source representations
    :param num_layers: number of layers
    :return: none
    """
    self.state = {}
    self.state["src"] = src
    self.state["cache"] = {}

    # device for multi-gpus
    device = str(memory_bank.device)
    # print(device)

    memory_keys = "memory_keys_" + device
    memory_values = "memory_values_" + device
    self_keys = "self_keys_" + device
    # print(self_keys)
    self_values = "self_values_" + device
    doc_keys = "document_keys_" + device
    doc_values = "document_values_" + device

    # build layer cache for each layer
    for l in range(num_layers):
        layer_cache = {
            memory_keys: None,
            memory_values: None,
            self_keys: None,
            self_values: None,
            doc_keys: None,
            doc_values: None,
        }
        # store in the cache in state
        self.state["cache"]["layer_{}".format(l)] = layer_cache


