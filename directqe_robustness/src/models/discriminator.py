# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.tensor_utils import tile_batch
from src.decoding.utils import tensor_gather_helper
from src.models.base import NMTModel
from src.modules.activation import GELU
from src.modules.attention import MultiHeadedAttention, MultiHeadedAttentionAligned
from src.modules.basic import BottleLinear as Linear
from src.modules.embeddings import Embeddings
from src.utils import nest
from src.utils.common_utils import Constants

PAD = Constants.PAD


class PositionwiseFeedForward(nn.Module):
    """
    A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        size (int): the size of input for the first-layer of the FFN.
        hidden_size (int): the hidden layer size of the second-layer
                          of the FNN.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, size, hidden_size, dropout=0.1, activation="relu"):  # size是模型的hidden_dim，这里的hidden_dim是ffnn的内部hidden_dim
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        # Save a little memory, by doing inplace.
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == "gelu":
            self.activation = GELU()
        else:
            raise ValueError

    def forward(self, x):  # x [batch_size, seq_len, hidden_dim]
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class Block(nn.Module):
    """
    The definition of block (sublayer) is formally introduced in Chen, Mia Xu et al.
    “The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation.” ACL (2018).

    A block is consist of a transform function (TF), a layer normalization (LN) and a residual connection with
    dropout (Drop-Res). There are two kinds of block, differing in the position of layer normalization:
        a): LN -> TF -> Drop-Res  (layer_norm_first is True)
        b): TF -> Drop-Res -> LN

    A block can return more than one output, but we only perform LN and Drop-Res on the first output.
    """

    def __init__(self, size, dropout, layer_norm_first=True):
        super().__init__()

        self.layer_norm_first = layer_norm_first

        self.layer_norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def _transform(self, *args, **kwargs):

        raise NotImplementedError

    def forward(self, x, *args, **kwargs):

        # 1. layer normalization
        if self.layer_norm_first:
            transform_input = self.layer_norm(x)
        else:
            transform_input = x

        # 2. transformation
        output = self._transform(transform_input, *args, **kwargs)

        # 3. dropout & residual add
        if not isinstance(output, tuple):
            output = x + self.dropout(output)
            if not self.layer_norm_first:
                output = self.layer_norm(output)
        else:
            output = (x + self.dropout(output[0]),) + output[1:]
            if not self.layer_norm_first:
                output = (self.layer_norm(output[0]),) + output[1:]

        return output


class SelfAttentionBlock(Block):

    def __init__(self, model_dim, head_count, dim_per_head=None, dropout=0.1, attn_dropout=0.1, layer_norm_firs=True):
        super().__init__(model_dim, dropout=dropout, layer_norm_first=layer_norm_firs)

        self.transform_layer = MultiHeadedAttention(model_dim=model_dim, head_count=head_count,
                                                    dim_per_head=dim_per_head, dropout=attn_dropout)

    def _transform(self, x, mask=None, self_attn_cache=None):
        return self.transform_layer(x, x, x, mask=mask, self_attn_cache=self_attn_cache)


class EncoderAttentionBlock(Block):

    def __init__(self, model_dim, head_count, dim_per_head=None, dropout=0.1, attn_dropout=0.1, layer_norm_first=True):
        super().__init__(model_dim, dropout=dropout, layer_norm_first=layer_norm_first)

        self.transform_layer = MultiHeadedAttentionAligned(model_dim=model_dim, head_count=head_count, \
                                            dim_per_head=dim_per_head, dropout=attn_dropout)
       
    def _transform(self, dec_hidden, context, mask=None, enc_attn_cache=None, align_matrix_pad=None, align_ratio=0):
        return self.transform_layer(context, context, dec_hidden, mask=mask, enc_attn_cache=enc_attn_cache, align_matrix_pad=align_matrix_pad, align_ratio=align_ratio)


class PositionwiseFeedForwardBlock(Block):

    def __init__(self, size, hidden_size, dropout=0.1, layer_norm_first=True, activation="relu"):
        super().__init__(size=size, dropout=dropout, layer_norm_first=layer_norm_first)

        self.transform_layer = PositionwiseFeedForward(size=size, hidden_size=hidden_size, dropout=dropout,
                                                       activation=activation)

    def _transform(self, x):
        return self.transform_layer(x)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1, layer_norm_first=True,
                 ffn_activation="relu"):
        super(EncoderLayer, self).__init__()

        self.slf_attn = SelfAttentionBlock(head_count=n_head, model_dim=d_model, dropout=dropout,
                                           dim_per_head=dim_per_head, layer_norm_firs=layer_norm_first)

        self.pos_ffn = PositionwiseFeedForwardBlock(size=d_model, hidden_size=d_inner_hid, dropout=dropout,
                                                    layer_norm_first=layer_norm_first, activation=ffn_activation)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        context, _, _ = self.slf_attn(enc_input, mask=slf_attn_mask)

        return self.pos_ffn(context)


class Encoder(nn.Module):

    def __init__(
            self, n_src_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1, dim_per_head=None,
            padding_idx=PAD, positional_embedding="sin", layer_norm_first=True, ffn_activation="relu"):
        super().__init__()

        self.scale = d_word_vec ** 0.5
        self.num_layers = n_layers
        self.layer_norm_first = layer_norm_first

        self.embeddings = Embeddings(num_embeddings=n_src_vocab,
                                     embedding_dim=d_word_vec,
                                     dropout=dropout,
                                     positional_embedding=positional_embedding
                                     )

        self.layer_stack = nn.ModuleList(
            [EncoderLayer(d_model=d_model, d_inner_hid=d_inner_hid, n_head=n_head, dropout=dropout,
                          dim_per_head=dim_per_head, layer_norm_first=layer_norm_first, ffn_activation=ffn_activation)
             for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src_seq, return_grads=False, return_embedding=False):
        # Word embedding look up
        batch_size, src_len = src_seq.size()

        if return_grads:
            emb, grads = self.embeddings(src_seq, return_grads=True)
        else:
            emb = self.embeddings(src_seq)

        enc_mask = src_seq.detach().eq(PAD)
        enc_slf_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, src_len, src_len)

        if not self.layer_norm_first:
            emb = self.layer_norm(emb)

        out = emb

        for i in range(self.num_layers):
            out = self.layer_stack[i](out, enc_slf_attn_mask)

        if self.layer_norm_first:
            out = self.layer_norm(out)

        if return_grads:
            return out, enc_mask, grads
        if return_embedding:
            return out, enc_mask, emb
        return out, enc_mask


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1, layer_norm_first=True,
                 ffn_activation="relu"):
        super(DecoderLayer, self).__init__()

        self.slf_attn = SelfAttentionBlock(head_count=n_head, model_dim=d_model, dropout=dropout,
                                           dim_per_head=dim_per_head, layer_norm_firs=layer_norm_first)
        self.ctx_attn = EncoderAttentionBlock(head_count=n_head, model_dim=d_model, dropout=dropout,
                                              dim_per_head=dim_per_head, layer_norm_first=layer_norm_first)

        self.pos_ffn = PositionwiseFeedForwardBlock(size=d_model, hidden_size=d_inner_hid,
                                                    layer_norm_first=layer_norm_first, activation=ffn_activation)

        self.dropout = nn.Dropout(dropout)

        self.adapter = Adapter(hidden_size=d_model, adapter_size=64)

        
    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None,
                enc_attn_cache=None, self_attn_cache=None, align_matrix_pad=None, align_ratio=0, 
                requires_adapter=False):
        # Args Checks
        input_batch, input_len, _ = dec_input.size()

        contxt_batch, contxt_len, _ = enc_output.size()

        query, _, self_attn_cache = self.slf_attn(dec_input, mask=slf_attn_mask, self_attn_cache=self_attn_cache)

        attn_values, attn_weights, enc_attn_cache = self.ctx_attn(query, enc_output, mask=dec_enc_attn_mask,
                                                                  enc_attn_cache=enc_attn_cache, align_matrix_pad=align_matrix_pad, align_ratio=align_ratio)

        output = self.pos_ffn(attn_values)

        if requires_adapter:
            output = self.adapter(output)

        return output, attn_weights, self_attn_cache, enc_attn_cache


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_tgt_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dim_per_head=None, dropout=0.1,
            positional_embedding="sin", layer_norm_first=True, padding_idx=PAD, ffn_activation="relu",
            d_feature=512):

        super(Decoder, self).__init__()

        self.n_head = n_head
        self.num_layers = n_layers
        self.d_model = d_model
        self.layer_norm_first = layer_norm_first

        self.embeddings = Embeddings(n_tgt_vocab, d_word_vec,
                                     dropout=dropout,
                                     positional_embedding=positional_embedding,
                                     padding_idx=padding_idx
                                     )

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_inner_hid=d_inner_hid, n_head=n_head, dropout=dropout,
                         dim_per_head=dim_per_head, layer_norm_first=layer_norm_first, ffn_activation=ffn_activation)
            for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model)

        self.feature2hidden = nn.Linear(d_feature, d_model)

        self._dim_per_head = dim_per_head

    @property
    def dim_per_head(self):
        if self._dim_per_head is None:
            return self.d_model // self.n_head
        else:
            return self._dim_per_head

    def forward(self, tgt_seq, enc_output, enc_mask, enc_attn_caches=None, self_attn_caches=None, 
                i_frac=1, return_grads=False, return_attn=False, align_matrix_pad=None, add_align=False, add_feature=False,
                align_ratio=0, enc_emb=None, requires_adapter=False):

        batch_size, tgt_len = tgt_seq.size()

        query_len = tgt_len
        key_len = tgt_len

        src_len = enc_output.size(1)

        # Run the forward pass of the TransformerDecoder.
        if return_grads:
            emb, grads = self.embeddings(tgt_seq, return_grads=True)
        else:
            emb = self.embeddings(tgt_seq)
        
        emb = emb * i_frac

        if not self.layer_norm_first:
            emb = self.layer_norm(emb)

        # 和对齐src emb相加
        if add_align:
            for mt_id, line in enumerate(align_matrix_pad):
                
                align_emb = torch.zeros_like(emb[0][mt_id])
                align_num = 0
                for src_id, st in enumerate(line):
                    if st == 1:
                        align_emb += enc_emb[0][src_id]
                        align_num += 1
                if align_num: align_emb /= align_num
                emb[0][mt_id] = emb[0][mt_id] * (1 - align_ratio) + align_emb * align_ratio
                
            
            align_matrix_pad = None

        align_matrix_pad_cpy = align_matrix_pad
        if add_feature:
            align_matrix_pad = None

        if self_attn_caches is not None:
            emb = emb[:, -1:].contiguous()
            query_len = 1

        # Decode mask
        dec_slf_attn_pad_mask = tgt_seq.detach().eq(PAD).unsqueeze(1).expand(batch_size, query_len, key_len)

        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask, 0)
        dec_enc_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, query_len, src_len)

        output = emb
        all_layer_attn = []
        new_self_attn_caches = []
        new_enc_attn_caches = []
        for i in range(self.num_layers):
            output, attn, self_attn_cache, enc_attn_cache \
                = self.layer_stack[i](output,
                                      enc_output,
                                      dec_slf_attn_mask,
                                      dec_enc_attn_mask,
                                      enc_attn_cache=enc_attn_caches[i] if enc_attn_caches is not None else None,
                                      self_attn_cache=self_attn_caches[i] if self_attn_caches is not None else None,
                                      align_matrix_pad=align_matrix_pad,
                                      align_ratio=align_ratio,
                                      requires_adapter=requires_adapter)

            all_layer_attn.append(attn)
            new_self_attn_caches += [self_attn_cache]
            new_enc_attn_caches += [enc_attn_cache]

        if add_feature:
            output_cat = torch.zeros_like(output)
            for mt_id, line in enumerate(align_matrix_pad_cpy):
                align_num = 0
                for src_id, st in enumerate(line):
                    if st == 1:
                        output_cat[0][mt_id] += enc_output[0][src_id]
                        align_num += 1
                if align_num: output_cat[0][mt_id] /= align_num
            
            output = torch.cat((output, output_cat), dim=-1)   # 和feature加在一起
            output = self.feature2hidden(output)   # 映射到原来的维度

        if self.layer_norm_first:
            output = self.layer_norm(output)   # 如果两个layer_norm的向量拼起来，还需要再过一次layer_norm吗

        if return_grads:
            return output, new_self_attn_caches, new_enc_attn_caches, grads
        elif return_attn:
            return output, new_self_attn_caches, new_enc_attn_caches, all_layer_attn
        else:
            return output, new_self_attn_caches, new_enc_attn_caches


class Classifier(nn.Module):

    def __init__(self, n_labels, hidden_size, shared_weight=None, padding_idx=-1, add_bias=False):
        super(Classifier, self).__init__()

        self.n_labels = n_labels
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.proj = Linear(self.hidden_size, self.n_labels, bias=add_bias)

        if shared_weight is not None:
            self.proj.linear.weight = shared_weight

    def _pad_2d(self, x):

        if self.padding_idx == -1:
            return x
        else:
            x_size = x.size()
            x_2d = x.view(-1, x.size(-1))

            mask = x_2d.new(1, x_2d.size(-1)).zero_()
            mask[0][self.padding_idx] = float('-inf')
            x_2d = x_2d + mask

            return x_2d.view(x_size)

    def forward(self, input, log_probs=True, sigmoid=False, mismatching=False, margin_loss=False):
        """
        input == > Linear == > LogSoftmax
        """

        logits = self.proj(input)

        if not sigmoid:
            logits = self._pad_2d(logits)

        if mismatching == True:
            return logits

        if sigmoid:
            return F.sigmoid(logits.squeeze(-1))
        if margin_loss:
            return logits
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


class Adapter(nn.Module):
    def __init__(self, hidden_size=512, adapter_size=64, adapter_initializer_range=1e-2):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        nn.init.normal_(self.down_project.weight, std=adapter_initializer_range)
        nn.init.zeros_(self.down_project.bias)

        self.activation = nn.ReLU(inplace=False)

        self.up_project = nn.Linear(adapter_size, hidden_size)
        nn.init.normal_(self.up_project.weight, std=adapter_initializer_range)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return hidden_states + up_projected


class Discriminator(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_layers=6, n_head=8, n_labels=3,
            d_word_vec=512, d_model=512, d_inner_hid=1024, d_feature=512, dim_per_head=None,
            dropout=0.1, tie_input_output_embedding=False, tie_source_target_embedding=False, padding_idx=PAD,
            layer_norm_first=True, positional_embedding="sin", generator_bias=False, ffn_activation="relu", **kwargs):
        super(Discriminator, self).__init__()

        self.encoder = Encoder(
            n_src_vocab, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout, dim_per_head=dim_per_head,
            padding_idx=padding_idx, layer_norm_first=layer_norm_first, positional_embedding=positional_embedding,
            ffn_activation=ffn_activation)

        self.decoder = Decoder(
            n_tgt_vocab, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model, d_feature=d_feature,
            d_inner_hid=d_inner_hid, dropout=dropout, dim_per_head=dim_per_head,
            padding_idx=padding_idx, layer_norm_first=layer_norm_first, positional_embedding=positional_embedding,
            ffn_activation=ffn_activation)

        self.hidden2hter = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module output shall be the same.'

        if tie_source_target_embedding:
            assert n_src_vocab == n_tgt_vocab, \
                "source and target vocabulary should have equal size when tying source&target embedding"
            self.encoder.embeddings.embeddings.weight = self.decoder.embeddings.embeddings.weight

        self.generator = Classifier(n_labels=n_labels, hidden_size=d_word_vec, padding_idx=PAD,
                                    add_bias=generator_bias)

        self.out_layer_norm = nn.LayerNorm(d_model)

    def forward(self, src_seq, tgt_seq, log_probs=True, i_frac=1, get_representation=False, get_hter_logits=False, return_grads=False, \
                return_attn=False, get_result_and_representation=False, align_matrix_pad = None, align_ratio = 0, add_align = False, requires_adapter=False, \
                add_feature=False, one_class=False, margin_loss=False):
        if return_grads:
            enc_output, enc_mask, enc_grads = self.encoder(src_seq, return_grads=True)
            dec_output, _, _, dec_grads = self.decoder(tgt_seq, enc_output, enc_mask, i_frac=i_frac, return_grads=True)
        elif return_attn:
            enc_output, enc_mask = self.encoder(src_seq)
            dec_output, _, _, ctx_attn = self.decoder(tgt_seq, enc_output, enc_mask, i_frac=i_frac, return_attn=True)
        elif add_align:
            enc_output, enc_mask, enc_emb = self.encoder(src_seq, return_embedding=True)
            dec_output, _, _ = self.decoder(tgt_seq, enc_output, enc_mask, i_frac=i_frac, align_matrix_pad = align_matrix_pad, add_align = add_align, \
                                            enc_emb = enc_emb, align_ratio=align_ratio, requires_adapter=requires_adapter)
        elif add_feature:
            enc_output, enc_mask = self.encoder(src_seq)
            dec_output, _, _ = self.decoder(tgt_seq, enc_output, enc_mask, i_frac=i_frac, align_matrix_pad = align_matrix_pad, \
                                            add_feature = add_feature, )

        else:
            enc_output, enc_mask = self.encoder(src_seq)
            dec_output, _, _ = self.decoder(tgt_seq, enc_output, enc_mask, i_frac=i_frac, align_matrix_pad = align_matrix_pad, align_ratio=align_ratio, \
                                            requires_adapter=requires_adapter)

        hter_output = dec_output[:, 1]        # [batch_size, d_model]
        if get_hter_logits == True:
            return hter_output
        hters = self.sigmoid(self.hidden2hter(hter_output))

        if get_representation == True:
            return dec_output

        if return_grads:
            return self.generator(dec_output, log_probs=log_probs)[:, 2:-1].contiguous(), hters.contiguous(), enc_grads, dec_grads
        if return_attn:
            return self.generator(dec_output, log_probs=log_probs)[:, 2:-1].contiguous(), hters.contiguous(), ctx_attn
        if get_result_and_representation:
            return self.generator(dec_output, log_probs=log_probs)[:, 2:-1].contiguous(), hters.contiguous(), dec_output[:, 2:-1]
        if one_class:
            return self.generator(dec_output, log_probs=log_probs, sigmoid=True)[:, 2:-1].contiguous(), hters.contiguous()
        if margin_loss:
            return self.generator(dec_output, log_probs=log_probs, margin_loss=True)[:, 2:-1].contiguous(), hters.contiguous()
            

        return self.generator(dec_output, log_probs=log_probs)[:, 2:-1].contiguous(), hters.contiguous()
