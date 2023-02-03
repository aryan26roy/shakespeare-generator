import torch
import torch.nn as nn
from torch.nn import functional as F

from hyperparameters import n_embd
from hyperparameters import block_size
from hyperparameters import dropout

class Head(nn.Module):
  def __init__(self,head_size):
    super().__init__()
    self.key  = nn.Linear(n_embd, head_size, bias=False) #each token describes itself
    self.query  = nn.Linear(n_embd, head_size, bias=False) #each token looks for specific tokens
    self.value  = nn.Linear(n_embd, head_size, bias=False) #transforms the input
    self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q @ k.transpose(-2,-1) * C**-0.5 #scaled multiplication of all quries with all keys (attention)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B,T,T), masking the future tokens
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out


class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim =-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  """One head of self attention"""
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))

  def forward(self,x):
    return self.net(x)

