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
import torch
import torch.nn as nn
from src.modules.rnn import RNN
from src.utils.common_utils import Constants
PAD = Constants.PAD
import src.utils.init as my_init


class QE(nn.Module):
    def __init__(self,
                 feature_size=2052,
                 hidden_size=512,
                 dropout=0.0,
                 **kwargs
                 ):
        super(QE, self).__init__()
        self.lstm = RNN(type="lstm", batch_first=True, input_size=feature_size, hidden_size=hidden_size,
                        bidirectional=True)
        self.w = nn.Linear(2 * hidden_size, 1)
        self.w_word = nn.Linear(2 * hidden_size, 1)
        my_init.default_init(self.w.weight)
        my_init.default_init(self.w_word.weight)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def cul_pre(self, context, mask):
        no_pad_mask = 1.0 - mask.float()
        ctx_mean = (context * no_pad_mask.unsqueeze(2)).sum(1) / no_pad_mask.unsqueeze(2).sum(1)
        pre = self.sigmoid(self.w(ctx_mean))
        return pre

    def forward(self, emb, x, level):
        """
        :param x: Input sequence.
            with shape [batch_size, seq_len, input_size]
        """
        x_mask = x.detach().eq(PAD)

        emb = self.dropout(emb)
        if level == 'sentence':
            ctx, _ = self.lstm(emb, x_mask)
            ctx = self.dropout(ctx)
            pre = self.cul_pre(ctx, x_mask)
        else:
            pre = self.w_word(emb)
            pre = self.dropout(pre)
            pre = pre[0, 1:-1, :]

        return pre