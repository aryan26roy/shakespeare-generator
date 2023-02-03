import torch
import torch.nn as nn
from torch.nn import functional as F
from models.attention import MultiHeadAttention
from models.feedforward import FeedForward

from hyperparameters import n_embd
from hyperparameters import n_head

class Block(nn.Module):
  """Transformer block"""
  def __init__(self, n_embed, n_head):
    super().__init__()
    head_size = n_embed//n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
