import torch, sys, numpy as np
import sentencepiece as spm
from train_tiny_gpt import TinyGPT, device, block_size, vocab_size
sp = spm.SentencePieceProcessor()
sp.Load("tokenizer.model")
model = TinyGPT().to(device)
model.load_state_dict(torch.load("tiny_gpt.pt", map_location=device))
model.eval()
def sample(prompt, max_tokens=50, temperature=1.0):
    ids = sp.EncodeAsIds(prompt)
    context = torch.tensor(ids[-block_size:], dtype=torch.long).unsqueeze(0).to(device)
    out = []
    for _ in range(max_tokens):
        logits = model(context)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        out.append(next_id)
        context = torch.cat([context, torch.tensor([[next_id]],device=device)], dim=1)
        if len(context[0])>block_size:
            context = context[:, -block_size:]
    return sp.DecodeIds(out)
if __name__=="__main__":
    prompt = sys.argv[1] if len(sys.argv)>1 else "hello"
    print(sample(prompt, max_tokens=40))
