
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        # input_dim = hidden_dim
        # output_dim = num_class
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj = nn.Linear(self.input_dim, self.output_dim)

    def forward(
        self,
        representation,
        return_score=False
    ):
        logits = self.proj(representation)
        if return_score:
            return logits
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
