import torch
batch_size= 64
block_size = 128
max_iters = 8000
eval_interval = 200
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 8
n_layer = 6
dropout = 0.2
vocab_size = 65
