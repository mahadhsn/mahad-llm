import sys
import json, os
from pathlib import Path
from training.tokenizer.byte_tokenizer import ByteLevelTokenizer

sys.path.append(str(Path(__file__).resolve().parents[2]))

tok = ByteLevelTokenizer(add_bos=False, add_eos=False)

def count_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return len(tok.encode(text))

root = Path("training/data/raw")
total = 0
for p in root.rglob("*.txt"):
    n = count_file(p)
    print(f"{p.name}: {n} tokens")
    total += n
print("TOTAL tokens (raw folder):", total)