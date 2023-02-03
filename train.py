import torch
import torch.nn as nn
from torch.nn import functional as F

from models.model import BigramLanguageModel
from utils.utilities import estimate_loss
from utils.utilities import get_batch
from utils.utilities import encode
from utils.utilities import decode

torch.manual_seed(1337)

from hyperparameters import device
from hyperparameters import learning_rate
from hyperparameters import max_iters
from hyperparameters import eval_interval

with open ('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()
  
chars = sorted(list(set(text)))
vocab_size = len(chars) #unique characters in the text

data = torch.tensor(encode(text), dtype=torch.long) #making all the data into a pytorch tensor
n = int(0.9*len(data))
train_data = data[:n] #train
val_data = data[n:] #validation

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #using adam

train_loss = []
val_loss = []
for iter in range(max_iters):

  if iter%eval_interval == 0:
    losses = estimate_loss(model, train_data, val_data)
    train_loss.append(losses['train'])
    val_loss.append(losses['val'])
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  xb, yb = get_batch('train', train_data, val_data)

  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
