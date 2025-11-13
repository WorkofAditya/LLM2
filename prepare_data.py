import os
import io
import sys
import gzip
import json
import numpy as np
from pathlib import Path
import sentencepiece as spm

dataset_dir = Path("data")
dataset_dir.mkdir(exist_ok=True)
greetings = [
"hi", "hello", "hey", "good morning", "good evening", "how are you?", "what's up?",
"i am fine", "nice to meet you", "bye", "see you", "good night"
]
with open(dataset_dir/"greetings.txt","w",encoding="utf8") as f:
    for s in greetings:
        f.write(s+"\n")
cornell_zip = dataset_dir/"cornell.zip"
if not (dataset_dir/"movie_lines.txt").exists():
    print("Please download Cornell Movie-Dialogs corpus into data/movie_lines.txt and data/movie_conversations.txt")
    sys.exit(1)
lines = {}
with open(dataset_dir/"movie_lines.txt","r",encoding="latin-1") as f:
    for l in f:
        parts = l.split(" +++$+++ ")
        if len(parts)>=5:
            lines[parts[0]] = parts[4].strip()
convos = []
with open(dataset_dir/"movie_conversations.txt","r",encoding="latin-1") as f:
    for l in f:
        parts = l.split(" +++$+++ ")
        if len(parts)>=4:
            import ast
            ids = ast.literal_eval(parts[3])
            txt = " __eot__ ".join([lines[i] for i in ids if i in lines])
            convos.append(txt)
all_text = "\n".join(convos[:5000]) + "\n" + open(dataset_dir/"greetings.txt","r",encoding="utf8").read()
with open("raw_corpus.txt","w",encoding="utf8") as f:
    f.write(all_text)
spm.SentencePieceTrainer.Train("--input=raw_corpus.txt --model_prefix=tokenizer --vocab_size=2000 --character_coverage=1.0 --model_type=bpe")
sp = spm.SentencePieceProcessor()
sp.Load("tokenizer.model")
tokens = sp.EncodeAsIds(all_text)
split = int(len(tokens)*0.9)
train = np.array(tokens[:split],dtype=np.uint16)
val = np.array(tokens[split:],dtype=np.uint16)
np.save("train.npy",train)
np.save("val.npy",val)
print("Saved train.npy and val.npy and tokenizer.model")
