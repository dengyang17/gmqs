import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models
import utils

# from fairseq import bleu
# from utils.reward_provider import CTRRewardProvider


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


class LabelSmoothingLoss(nn.Module):
    """ Label smoothing loss """
    def __init__(self, device, label_smoothing, vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()
        self.size = vocab_size
        self.device = device
        self.smoothing_value = label_smoothing / (vocab_size - 2)
        one_hot = torch.full((vocab_size,), self.smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        real_size = output.size(1)
        if real_size > self.size:
            real_size -= self.size
        else:
            real_size = 0

        model_prob = self.one_hot.repeat(target.size(0), 1)
        if real_size > 0:
            ext_zeros = torch.full((model_prob.size(0), real_size), self.smoothing_value).to(self.device)
            model_prob = torch.cat((model_prob, ext_zeros), -1)

        #model_prob = self.one_hot.repeat(target.size(0), 1)
        #if extra_zeros is not None:
        #    extra_zeros = extra_zeros.contiguous().view(-1, extra_zeros.size(2)) 
        #    extra_zeros += self.smoothing_value
        #    model_prob = torch.cat((model_prob, extra_zeros), -1)

        #output = F.log_softmax(output, dim=-1)
        #model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class tensor2tensor(nn.Module):
    """ transformer model """
    def __init__(self, config, device, use_attention=True,
                 encoder=None, decoder=None,
                 src_padding_idx=0, tgt_padding_idx=0,
                 label_smoothing=0, tgt_vocab=None):
        """
        Initialization of variables and functions
        :param config: configuration
        :param use_attention: use attention or not, consistent with seq2seq
        :param encoder: encoder
        :param decoder: decoder
        :param src_padding_idx: source padding index
        :param tgt_padding_idx: target padding index
        :param label_smoothing: ratio for label smoothing
        :param tgt_vocab: target vocabulary
        """
        super(tensor2tensor, self).__init__()

        self.config = config
        self.device = device

        # pretrained encoder or not
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = models.TransformerEncoder(
                config, padding_idx=src_padding_idx)

        self.condition_context_attn = models.BiAttention(config.hidden_size, config.dropout)
        self.bi_attn_transform = nn.Linear(config.hidden_size * 4, config.hidden_size)


        if config.gnn == 'rgcn':
            self.infer = models.RGCN(config)
        elif config.gnn == 'rgat':
            self.infer = models.RGAT(config)
        elif config.gnn == 'transformer':
            self.infer = models.TransformerInterEncoder(config)
        else:
            self.infer = models.Classifier(config)

        tgt_embedding = self.encoder.embedding
        # pretrained decoder or not
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = models.TransformerDecoder(
                config, tgt_embedding=tgt_embedding, padding_idx=tgt_padding_idx)
        # log softmax should specify dimension explicitly
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.use_cuda = config.use_cuda
        self.config = config
        self.label_smoothing = label_smoothing
        if self.label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(device,
                label_smoothing, config.vocab_size,
                ignore_index=tgt_padding_idx)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=utils.PAD)
        if config.use_cuda:
            self.criterion.to(self.device)
        self.compute_score = nn.Linear(
            config.hidden_size, config.vocab_size)
        
        self.ext_loss = torch.nn.BCELoss(reduction='none')

        self.padding_idx = tgt_padding_idx

    def compute_loss(self, scores, targets):
        """
        loss computation
        :param scores: predicted scores
        :param targets: targets
        :return: loss
        """
        scores = scores.contiguous().view(-1, scores.size(2))   #[batch*len, vocab]
        loss = self.criterion(scores, targets.contiguous().view(-1))
        return loss
    
    def pointer_network(self, outputs, attn, doc_attn, pointers, src_extend_ids, max_ext_len):
        bsz, output_len, _ = outputs.size()
        vocab_dist = F.softmax(outputs, dim=-1)
        if max_ext_len > 0:
            extra_zeros = Variable(torch.zeros(bsz, output_len, max_ext_len)).to(self.device)
            vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1) 
        vocab_dist = vocab_dist * pointers[:,:,0].unsqueeze(-1)
        src_len = attn.size(2)
        doc_len = doc_attn.size(2)
        vocab_dist = vocab_dist.scatter_add(2, src_extend_ids[:,:src_len].unsqueeze(1).expand(bsz, output_len, src_len), attn * pointers[:,:,1].unsqueeze(-1))
        vocab_dist = vocab_dist.scatter_add(2, src_extend_ids[:,src_len:].unsqueeze(1).expand(bsz, output_len, doc_len), doc_attn * pointers[:,:,2].unsqueeze(-1))
        return vocab_dist

    def forward(self, src, dec, targets, src_extend_ids, tgt_extend_ids, max_ext_len, ext_label, adjs):
        """
        run transformer
        :param src: source input
        :param src_len: source length
        :param dec: decoder input
        :param targets: target
        :return: dictionary of loss, reward, etc., output scores
        """
        return_dict = {}

        ds0, ds1, ds2 = src.size()
        src = src.view(ds0, -1)
        src_extend_ids = src_extend_ids.view(ds0, -1)

        contexts = self.encoder(src)  # batch, len, size

        sent_vec = contexts.view(ds0, ds1, ds2, -1)
        mask_label = (ext_label != -1)
        sent_vec = torch.mean(sent_vec, 2)#[:,1:]

        if self.config.gnn in ['rgcn','rgat']:
            sent_vec, sent_scores = self.infer(sent_vec, adjs, mask_label)
        else:
            sent_vec, sent_scores = self.infer(sent_vec, mask_label)
        loss = self.ext_loss(sent_scores, ext_label.float())
        loss = (loss * mask_label.float()).sum()

        contexts = contexts.view(ds0, ds1, ds2, -1)
        question_contexts = contexts[:, 0].view(ds0, ds2, -1)
        document_contexts = contexts[:, 1:].view(ds0, (ds1-1)*ds2, -1)

        models.transformer.init_state(self.decoder, src, contexts, self.decoder.num_layers)

        sent_pad_mask = (ext_label == -1)
        outputs, attn_weights, doc_attn, pointers = self.decoder(dec, question_contexts, document_contexts, sent_vec, sent_pad_mask) # [batch, len, size]

        scores = self.compute_score(outputs) # [batch, len, vocab]
        if pointers is not None:
            scores = self.pointer_network(scores, attn_weights, doc_attn, pointers, src_extend_ids, max_ext_len)
            scores = torch.log(scores.clamp(min=1e-8))
            return_dict["mle_loss"] = self.compute_loss(scores, tgt_extend_ids) + 0.5*loss
        else:
            scores = F.log_softmax(scores, dim=-1)
            return_dict["mle_loss"] = self.compute_loss(scores, targets) + 0.5*loss
        return return_dict, scores

    def sample(self, src, src_extend_ids, max_ext_len, ext_label, adjs):
        """
        Greedy sampling for inference
        :param src: source input
        :param src_len: source length
        :return: sampled ids and attention alignment
        """
        bos = torch.ones(src.size(0)).long().fill_(utils.BOS).to(self.device)   # [batch]
        
        ds0, ds1, ds2 = src.size()
        src = src.view(ds0, -1)
        src_extend_ids = src_extend_ids.view(ds0, -1)

        contexts = self.encoder(src)  # batch, len, size

        sent_vec = contexts.view(ds0, ds1, ds2, -1)
        mask_label = (ext_label != -1)
        sent_vec = torch.mean(sent_vec, 2)

        if self.config.gnn in ['rgcn','rgat']:
            sent_vec, sent_scores = self.infer(sent_vec, adjs, mask_label)
        else:
            sent_vec, sent_scores = self.infer(sent_vec, mask_label)

        contexts = contexts.view(ds0, ds1, ds2, -1)
        question_contexts = contexts[:, 0].view(ds0, ds2, -1)
        document_contexts = contexts[:, 1:].view(ds0, (ds1-1)*ds2, -1)

        models.transformer.init_state(self.decoder, src, contexts, self.decoder.num_layers)

        sent_pad_mask = (ext_label == -1)

        # sequential generation
        inputs, outputs, attn_matrix = [bos], [], []
        for i in range(self.config.max_time_step):
            output, attn_weights, doc_attn, pointers = self.decoder(inputs[i].unsqueeze(1), question_contexts, document_contexts, sent_vec, sent_pad_mask, step=i) # [batch, len, size]
            output = self.compute_score(output)  # [batch, 1, size]
            if pointers is not None:
                output = self.pointer_network(output, attn_weights, doc_attn, pointers, src_extend_ids, max_ext_len)
            predicted = output.squeeze(1).max(1)[1]    # [batch]
            latest_tokens = [t if t < self.config.vocab_size else utils.UNK for t in predicted]
            latest_tokens = torch.LongTensor(latest_tokens).to(self.device) 
            inputs.append(latest_tokens)
            outputs.append(predicted)
            attn_matrix.append(attn_weights.squeeze(1)) #[batch, k_len]
        outputs = torch.stack(outputs)  # [batch, len]
        # select by the indices along the dimension of batch
        sample_ids = outputs.t().tolist()

        attn_matrix = torch.stack(attn_matrix)  # [batch, len, k_len]
        alignments = attn_matrix.max(2)[1].t().tolist() # [batch, len]
        
        return sample_ids, alignments, sent_scores

    def beam_sample(self, src, src_extend_ids, max_ext_len, min_dec_len, ext_label, adjs, beam_size=1, eval_=False):
        """
        beam search
        :param src: source input
        :param src_len: source length
        :param beam_size: beam size
        :param eval_: evaluation or not
        :return: prediction, attention max score and attention weights
        """
        batch_size = src.size(0)

        ds0, ds1, ds2 = src.size()
        src = src.view(ds0, -1)
        src_extend_ids = src_extend_ids.view(ds0, -1)

        contexts = self.encoder(src)  # batch, len, size

        sent_vec = contexts.view(ds0, ds1, ds2, -1)
        mask_label = (ext_label != -1)
        sent_vec = torch.mean(sent_vec, 2)

        if self.config.gnn in ['rgcn','rgat']:
            sent_vec, sent_scores = self.infer(sent_vec, adjs, mask_label)
        else:
            sent_vec, sent_scores = self.infer(sent_vec, mask_label)

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(batch_size, beam_size, -1)

        beam = [models.Beam(beam_size, n_best=1,
                            cuda=self.use_cuda, length_norm=self.config.length_norm, minimum_length=min_dec_len)
                for __ in range(batch_size)]    # [batch, beam]

        contexts = tile(contexts, beam_size, 0) # [batch*beam, len, size]
        src = tile(src, beam_size, 0)   # [batch*beam, len]
        src_extend_ids = tile(src_extend_ids, beam_size, 0)

        contexts = contexts.view(ds0*beam_size, ds1, ds2, -1)
        question_contexts = contexts[:, 0].view(ds0*beam_size, ds2, -1)
        document_contexts = contexts[:, 1:].view(ds0*beam_size, (ds1-1)*ds2, -1)

        # self.decoder.init_state(src, contexts)
        models.transformer.init_state(self.decoder, src, contexts, self.decoder.num_layers)

        sent_pad_mask = (ext_label == -1)
        sent_vec = tile(sent_vec, beam_size, 0)
        sent_pad_mask = tile(sent_pad_mask, beam_size, 0)

        # sequential generation
        for i in range(self.config.max_time_step):
            # finish beam search
            if all((b.done() for b in beam)):
                break

            inp = torch.stack([torch.LongTensor([t if t < self.config.vocab_size else utils.UNK for t in b.getCurrentState()]).to(self.device) for b in beam])
            inp = inp.view(-1, 1)   # [batch*beam, 1]

            output, attn, doc_attn, pointers = self.decoder(inp, question_contexts, document_contexts, sent_vec, sent_pad_mask, step=i) # [batch*beam, len, size]
            state = None
            output = self.compute_score(output)  # [batch*beam, 1, size]
            if pointers is not None:
                output = self.pointer_network(output, attn, doc_attn, pointers, src_extend_ids, max_ext_len)
                output = unbottle(torch.log(output.squeeze(1).clamp(min=1e-8)))
            else:
                output = unbottle(self.log_softmax(output.squeeze(1))) # [batch, beam, size]
            attn = unbottle(attn.squeeze(1))    # [batch, beam, k_len]

            select_indices_array = []
            # scan beams in a batch
            for j, b in enumerate(beam):
                # update each beam
                b.advance(output[j, :], attn[j, :]) # batch index
                select_indices_array.append(
                    b.getCurrentOrigin() + j * beam_size)
            select_indices = torch.cat(select_indices_array)
            self.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        allHyps, allScores, allAttn = [], [], []
        if eval_:
            allWeight = []

        for j in range(batch_size):
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            if eval_:
                weight = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
                if eval_:
                    weight.append(att)
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])
            if eval_:
                allWeight.append(weight[0])
        if eval_:
            return allHyps, allAttn, allWeight

        return allHyps, allAttn, sent_scores
