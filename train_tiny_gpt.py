import math, time, os, sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from tqdm import trange
device = "cuda" if torch.cuda.is_available() else "cpu"
train_tokens = np.load("train.npy")
val_tokens = np.load("val.npy")
train_data = torch.from_numpy(train_tokens).long().to(device)
val_data = torch.from_numpy(val_tokens).long().to(device)
batch_size = 16
block_size = 128
vocab_size = 2000
n_layer = 2
n_head = 4
n_embd = 128
dropout = 0.1
class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.size()
        k = self.key(x).view(B,T,-1)
        q = self.query(x).view(B,T,-1)
        wei = q @ k.transpose(-2,-1) * (1.0/ math.sqrt(k.size(-1)))
        wei = wei.masked_fill(self.tril[:T,:T]==0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
class MultiHead(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHead(n_embd, n_head)
        self.ln1 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
    def forward(self, idx, targets=None):
        B,T = idx.size()
        tok = self.tok_emb(idx)
        x = tok + self.pos_emb[:,:T,:]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))
        return logits, loss
model = TinyGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
def get_batch(split):
    data = train_data if split=="train" else val_data
    ix = torch.randint(0, data.size(0)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train","val"]:
        losses = []
        for _ in range(5):
            xb,yb = get_batch(split)
            with torch.no_grad():
                _,loss = model(xb, yb)
            losses.append(loss.item())
        out[split]=sum(losses)/len(losses)
    model.train()
    return out
iters = 2000
print_every = 100
for it in range(iters):
    xb,yb = get_batch("train")
    logits,loss = model(xb,yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if it%print_every==0:
        losses = estimate_loss()
        print(f"iter {it} train {losses['train']:.4f} val {losses['val']:.4f}")
        torch.save(model.state_dict(),"tiny_gpt.pt")
