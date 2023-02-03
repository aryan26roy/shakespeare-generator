import torch
from models.model import BigramLanguageModel
from hyperparameters import device
from utils.utilities import encode
from utils.utilities import decode


model = BigramLanguageModel()
model_save_name = 'weights/transformer_final.pt'
model.load_state_dict(torch.load(model_save_name))
m = model.to(device)
print("Enter Prompt:\n")
prompt = input()
length = len(prompt)
data = torch.tensor(encode(prompt), dtype=torch.long, device = device)
data = data.reshape(1,length)
print("\n========================================================================\n")
print(decode(m.generate(idx = data, max_new_tokens=500)[0].tolist())) 
