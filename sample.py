import torch
import sys
import sentencepiece as spm
import torch.nn as nn
import torch.nn.functional as F
import math

# constants (must match training)
vocab_size = 2000
block_size = 128
n_layer = 2
n_head = 4
n_embd = 128
dropout = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"

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
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (1.0/math.sqrt(k.size(-1)))
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
    def forward(self, idx):
        B,T = idx.size()
        tok = self.tok_emb(idx)
        x = tok + self.pos_emb[:,:T,:]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

sp = spm.SentencePieceProcessor()
sp.Load("tokenizer.model")

model = TinyGPT().to(device)
model.load_state_dict(torch.load("tiny_gpt.pt", map_location=device))
model.eval()

def sample_text(prompt, max_tokens=40, temperature=1.0):
    ids = sp.EncodeAsIds(prompt)
    context = torch.tensor(ids[-block_size:], dtype=torch.long).unsqueeze(0).to(device)
    out = []
    for _ in range(max_tokens):
        logits = model(context)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        out.append(next_id)
        context = torch.cat([context, torch.tensor([[next_id]], device=device)], dim=1)
        if len(context[0]) > block_size:
            context = context[:, -block_size:]
    return sp.DecodeIds(out)

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "hello"
    print(sample_text(prompt, max_tokens=40, temperature=1.0))
