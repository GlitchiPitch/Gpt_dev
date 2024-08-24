import torch

from content import text
from config import *
from network import BigramLangModel

chars = sorted(list(set(text)))
vocab_size = len(chars)

torch.manual_seed(1337)

stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i : i + block_size] for i in ix])
  y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
  return x, y

class BatchNorm1d:

  def __init__(self, dim, eps=1e-5):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):

    xmean = x.mean(1, keepdim=True)
    xvar = x.var(1, keepdim=True)
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]
  
model = BigramLangModel(vocab_size=vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iter)
    for k in range(10): #eval_iter
      x, y = get_batch(split)
      logits, loss = model(x, y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

for iter in range(max_iter):

  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}, train loss {losses['train']:.4f} val loss {losses['val']:.4f}")

  xb, yb = get_batch('train')

  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

def main():
    context = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))

if __name__ == "__main__":
    main()