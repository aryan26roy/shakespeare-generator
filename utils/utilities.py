import torch
import torch.nn as nn
from torch.nn import functional as F

from hyperparameters import eval_iters
from hyperparameters import block_size
from hyperparameters import batch_size
from hyperparameters import device

with open ('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()
  
chars = sorted(list(set(text)))

stoi = { ch:i for i,ch in enumerate(chars) } #can use sentencepiece here as well (or any other subword tokenizer)
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

def get_batch(split, train_data, val_data):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data)-block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size]for i in ix])
  y = torch.stack([data[i+1:i+block_size+1]for i in ix])
  x,y = x.to(device), y.to(device)
  return x,y

@torch.no_grad()
def estimate_loss(model,train_data, val_data):
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X,Y = get_batch(split,train_data, val_data)
      logits, loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out
