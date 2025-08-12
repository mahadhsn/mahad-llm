import json
import argparse
import random
from pathlib import Path

from training.tokenizer.byte_tokenizer import ByteLevelTokenizer, BOS, EOS

def load_jsonl(path):
    """Yield the 'text' field from each JSONL line."""
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if text:
                yield text

def text_to_chunks(text, token, max_len):
    """
    Encode text -> token ids (optionally with BOS/EOS) and
    slice into non-overlapping chunks of length max_len.
    We keep only full chunks here to keep the demo simple.
    """
    ids = token.encode(text, add_special=(token.add_bos or token.add_eos))
    chunks = []
    i = 0
    while i + max_len <= len(ids):
        chunks.append(ids[i: i + max_len])
        i += max_len
    return chunks

def make_batch(jsonl_path, seq_len, batch_size, add_bos, add_eos, seed=42):
    """
    Build ONE mini-batch (X, Y) where:
      X: list[list[int]] input_ids, each length (seq_len - 1)
      Y: list[list[int]] target_ids (same length), shifted by 1
    """
    random.seed(seed)
    token = ByteLevelTokenizer(add_bos=add_bos, add_eos=add_eos)
    
    texts = list(load_jsonl(jsonl_path))
    if not texts:
        print("Error")
        return
    random.shuffle(texts)
    
    X, Y = [], []
    
    for text in texts:
        for seq in text_to_chunks(text, token, max_len=seq_len):
            inp = seq[:-1]
            tgt = seq[1:]
            X.append(inp)
            Y.append(tgt)
            if len(X) == batch_size:
                return X, Y, token
            
def preview_batch(X, Y, token, n_preview=2, n_chars=80):
    """
    Print short previews so you can *see* the shift-by-one relationship.
    """
    n = min(n_preview, len(X))
    print(f"Got batch of {len(X)} examples; showing {n} preview(s).\n")
    for i in range(n):
        x, y = X[i], Y[i]
        print(f"Example {i}")
        print(" input_ids len:", len(x), " target_ids len:", len(y))
        # Show the first ~12 token IDs to visualize the shift
        print(" input ids head:", x[:12])
        print(" target ids head:", y[:12])
        # Normal decoded previews (specials are hidden)
        x_text = token.decode(x)[:n_chars].replace("\n", "\\n")
        y_text = token.decode(y)[:n_chars].replace("\n", "\\n")
        print(" input preview:", x_text)
        print(" target preview:", y_text)
        print("-" * 60)


def main():
    p = argparse.ArgumentParser(description="Build a tiny demo batch from JSONL.")
    p.add_argument("--jsonl", required=True, help="Path to JSONL (e.g., training/data/clean/mahad_train.jsonl)")
    p.add_argument("--seq-len", type=int, default=128, help="Chunk length (e.g., 128, 256)")
    p.add_argument("--batch-size", type=int, default=2, help="Number of examples to collect")
    p.add_argument("--add-bos", action="store_true", help="Insert BOS token at start")
    p.add_argument("--add-eos", action="store_true", help="Insert EOS token at end")
    args = p.parse_args()

    X, Y, tok = make_batch(
        jsonl_path=args.jsonl,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
    )
    if not X:
        raise SystemExit("No sequences created. Try a smaller --seq-len or add more data.")
    preview_batch(X, Y, tok)


if __name__ == "__main__":
    main()