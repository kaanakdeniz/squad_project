"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, *args):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAFCharEmbed(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAFCharEmbed, self).__init__()
        self.word_emb = layers.WordEmbedding(word_vectors=word_vectors,
                                             hidden_size=hidden_size,
                                             drop_prob=drop_prob)

        self.char_emb = layers.CharEmbedding(char_vectors=char_vectors,
                                             hidden_size=hidden_size,
                                             drop_prob=drop_prob,
                                             kernel_height=3)

        self.highway = layers.HighwayEncoder(num_layers=2, hidden_size=2 * hidden_size)

        self.highway_proj = nn.Linear(2 * hidden_size, hidden_size)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, *args):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        cw_emb = self.word_emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        qw_emb = self.word_emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        cc_emb = self.char_emb(cc_idxs)         # (batch_size, c_len, hidden_size)
        qc_emb = self.char_emb(qc_idxs)         # (batch_size, q_len, hidden_size)

        c_emb_cat = torch.cat((cw_emb, cc_emb), 2)  # (batch_size, c_len, 2 * hidden_size)
        q_emb_cat = torch.cat((qw_emb, qc_emb), 2)  # (batch_size, q_len, 2 * hidden_size)

        c_emb = self.highway_proj(self.highway(c_emb_cat))  # (batch_size, c_len, hidden_size)
        q_emb = self.highway_proj(self.highway(q_emb_cat))  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class BiDAFSelfAttention(nn.Module):
    def __init__(self, word_vectors, hidden_size, att_type=None, drop_prob=0., **kwargs):
        super(BiDAFSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)
        self.self_att = init_attention(att_type, hidden_size, drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, *args):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        self_att = self.self_att(att)   # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(self_att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(self_att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAFCharEmbedSelfAttention(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, att_type=None, drop_prob=0., **kwargs):
        super(BiDAFCharEmbedSelfAttention, self).__init__()
        self.word_emb = layers.WordEmbedding(word_vectors=word_vectors,
                                             hidden_size=hidden_size,
                                             drop_prob=drop_prob)

        self.char_emb = layers.CharEmbedding(char_vectors=char_vectors,
                                             hidden_size=hidden_size,
                                             drop_prob=drop_prob,
                                             kernel_height=3)

        self.highway = layers.HighwayEncoder(num_layers=2, hidden_size=2 * hidden_size)

        self.highway_proj = nn.Linear(2 * hidden_size, hidden_size)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)


        self.self_att = init_attention(att_type, hidden_size, drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, *args):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        cw_emb = self.word_emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        qw_emb = self.word_emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        cc_emb = self.char_emb(cc_idxs)         # (batch_size, c_len, hidden_size)
        qc_emb = self.char_emb(qc_idxs)         # (batch_size, q_len, hidden_size)

        c_emb_cat = torch.cat((cw_emb, cc_emb), 2)  # (batch_size, c_len, 2 * hidden_size)
        q_emb_cat = torch.cat((qw_emb, qc_emb), 2)  # (batch_size, q_len, 2 * hidden_size)

        c_emb = self.highway_proj(self.highway(c_emb_cat))  # (batch_size, c_len, hidden_size)
        q_emb = self.highway_proj(self.highway(q_emb_cat))  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        self_att = self.self_att(att)

        mod = self.mod(self_att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(self_att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

def init_attention(att_type, hidden_size, drop_prob):
    self_att = None
    if att_type == 'multiplicative':
        self_att = layers.MultiplicativeSelfAttention(input_size=8 * hidden_size,
                                                            drop_prob=drop_prob)
    elif att_type == 'gated_mult':
        self_att = layers.GatedMultiplicativeSelfAttention(input_size=8 * hidden_size,
                                                                hidden_size=4 * hidden_size,
                                                                drop_prob=drop_prob)
    elif att_type == 'additive':
        self_att = layers.AdditiveSelfAttention(input_size=8 * hidden_size,
                                                        att_dim=hidden_size,
                                                        drop_prob=drop_prob)
    elif att_type == 'gated_add':
        self_att = layers.GatedAdditiveSelfAttention(input_size=8 * hidden_size,
                                                            att_dim=hidden_size,
                                                            hidden_size=4 * hidden_size,
                                                            drop_prob=drop_prob)
    else:
        raise ValueError(f'{att_type} attention has not been implemented')

    return self_att


def init_model(name, split, **kwargs):
    name = name.lower()
    print(f'Initializing model: {name}')
    if name == 'bidaf':
        return BiDAF(word_vectors=kwargs['word_vectors'],
                     hidden_size=kwargs['hidden_size'],
                     drop_prob=kwargs['drop_prob'] if split == 'train' else 0)
    elif name == 'char_emb':
        return BiDAFCharEmbed(word_vectors=kwargs['word_vectors'],
                              char_vectors=kwargs['char_vectors'],
                              hidden_size=kwargs['hidden_size'],
                              drop_prob=kwargs['drop_prob'] if split == 'train' else 0)
    elif name == 'attention':
        attention_type = kwargs['attention']
        print(f'Using {attention_type} attention')
        return BiDAFSelfAttention(word_vectors=kwargs['word_vectors'],
                                  hidden_size=kwargs['hidden_size'], att_type=attention_type,
                                  drop_prob=kwargs['drop_prob'] if split == 'train' else 0)
    elif name == 'attention_charemb':
        attention_type = kwargs['attention']
        print(f'Using {attention_type} attention')
        return BiDAFCharEmbedSelfAttention(word_vectors=kwargs['word_vectors'],
                                           char_vectors=kwargs['char_vectors'],
                                           hidden_size=kwargs['hidden_size'], att_type=attention_type,
                                           drop_prob=kwargs['drop_prob'] if split == 'train' else 0)
    raise ValueError(f'No model named {name}')