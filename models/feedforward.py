import torch
import torch.nn as nn
from torch.nn import functional as F
from models.attention import MultiHeadAttention

from hyperparameters import n_embd
from hyperparameters import dropout

class FeedForward(nn.Module):
  """One head of self attention"""
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))

  def forward(self,x):
    return self.net(x)
