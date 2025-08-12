import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from training.scripts.make_example_batch import make_batch
from training.tokenizer.byte_tokenizer import ByteLevelTokenizer
from training.model.minigpt import MiniLM

def main():
    
    jsonl_path = "training/data/clean/mahad_train.jsonl"
    seq_len = 128
    batch_size = 2
    
    X_list, Y_list, tok = make_batch(
        jsonl_path=jsonl_path,
        seq_len=seq_len,
        batch_size=batch_size,
        add_bos=True,
        add_eos=True,
    )
    
    X = torch.tensor(X_list, dtype=torch.long)
    Y = torch.tensor(Y_list, dtype=torch.long)
    
    B, T = X.shape
    vocab_size = 259
    
    model = MiniLM(vocab_size=vocab_size, d_model=256, max_seq_len=T)
    logits = model(X)
    
    logits_flat = logits.reshape(B * T, vocab_size)
    targets_flat = Y.reshape(B * T)
    
    loss = nn.CrossEntropyLoss()(logits_flat, targets_flat)
    print("Logits shape: ", logits.shape)
    print("Loss: ", float(loss.item()))
    
if __name__ == "__main__":
    main()