import torch
import torch.nn as nn
from torch.nn import functional as F
from models.block import Block

from hyperparameters import vocab_size
from hyperparameters import n_embd
from hyperparameters import block_size
from hyperparameters import n_head
from hyperparameters import n_layer
from hyperparameters import device

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # a table for looking up the next possible token
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(Block(n_embd,n_head=4),Block(n_embd,n_head=4),Block(n_embd,n_head=4), nn.LayerNorm(n_embd))
    self.blocks = nn.Sequential(*[Block(n_embd, n_head= n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets = None):
    B, T = idx.shape
    tok_embd = self.token_embedding_table(idx) # output is in the format (B,T,C) or (batch_size, block_size, vocab_size)
    pos_embd = self.position_embedding_table(torch.arange(T,device=device)) # (T,C)
    x = tok_embd + pos_embd #(B,T,C)
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # (B,T, vocab_size)

    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C) # reshaping since the channel is expected to be the second dimension by pytorch's implementation of cross_entropy
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]# cropping input
      logits, loss = self(idx_cond) # goes to the forward method
      logits = logits[:, -1, :] #getting the row from the embedding table of the last charachter in the input sequence
      probs = F.softmax(logits, dim= -1) #making a probability distribution
      idx_next = torch.multinomial(probs, num_samples=1) #sampling the most likely next charachter from the said distribution
      idx = torch.cat((idx, idx_next), dim = 1) #adding the prediction to the input
    return idx
