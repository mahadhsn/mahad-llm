import os, json, re, random
from pathlib import Path

config_path = Path("training/data/config/names_ignore.json")
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as file:
        cfg = json.load(file)
        manual_names = set(n.lower() for n in cfg.get("manual_names", []))
else:
    print("ERROR: COULDN'T LOAD JOSN")
    
ALL_NAMES = manual_names

RAW = Path("training/data/raw")
OUT = Path("training/data/clean")

# --- Cleaning patterns ---
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b")
HANDLE_RE = re.compile(r"(?<!\w)@\w+")
EMAIL_RE  = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE    = re.compile(r"https?://\S+")
NAME_RE = re.compile(r"\b(" + "|".join(re.escape(n) for n in ALL_NAMES) + r")\b", re.IGNORECASE)

# Common WhatsApp/system artifacts to drop entirely if a line contains them
SYSTEM_MARKERS = [
    "Messages and calls are end-to-end encrypted",
    "This message was deleted",
    "You deleted this message",
    "<Media omitted>",
    "Missed voice call",
    "Missed video call",
    "image omitted",
    "video omitted",
    "sticker omitted",
    "GIF omitted",
    "<This message was edited>",
]

def scrub_names(text):
    return NAME_RE.sub("<name>", text)

def normalize_spaces(s):
    s = s.replace("\r\n", "\n")
    # Collapse spaces/tabs but keep newlines
    s = re.sub(r"[^\S\n]+", " ", s)  # whitespace except newline -> single space
    # Collapse multiple blank lines to a single newline
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def clean_text(s: str) -> str:
    """Light scrubbing: drop system artifacts and replace sensitive items with placeholders."""
    # Remove lines containing system markers (WhatsApp export noise)
    kept = []
    for line in s.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if any(marker.lower() in line_stripped.lower() for marker in SYSTEM_MARKERS):
            continue
        kept.append(line_stripped)
    s = "\n".join(kept)  # keep soft sentence boundaries

    # Replace URLs / emails / phones / @handles with placeholders
    s = URL_RE.sub("<url>", s)
    s = EMAIL_RE.sub("<email>", s)
    s = PHONE_RE.sub("<phone>", s)
    s = HANDLE_RE.sub("<handle>", s)

    # Collapse whitespace
    return normalize_spaces(s)

def split_into_utterances(s: str, max_len: int = 160, group: bool = False):
    """Split into sentence-like chunks.
    If group=False (default): return each sentence as its own segment (trimmed).
    If group=True: group adjacent sentences up to max_len characters.
    We also treat newlines as hard boundaries in addition to . ! ?
    """
    # First split on newlines to respect chat-style breaks
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    sentences = []
    sent_re = re.compile(r"(?<=[.!?])\s+")
    for ln in lines:
        parts = sent_re.split(ln)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)

    if not sentences:
        return []

    if not group:
        # Return one sentence per segment, trimmed to <= max_len (hard cut if needed)
        out = []
        for sent in sentences:
            if len(sent) <= max_len:
                out.append(sent)
            else:
                # Hard wrap long sentences into chunks of size max_len
                i = 0
                while i < len(sent):
                    out.append(sent[i:i+max_len].strip())
                    i += max_len
        return out

    # group=True: accumulate sentences until we hit the budget
    chunks, buf, cur = [], [], 0
    for sent in sentences:
        # +1 for a space/newline between sentences when joining
        add = len(sent) + (1 if buf else 0)
        if cur + add <= max_len:
            buf.append(sent)
            cur += add
        else:
            if buf:
                chunks.append("\n".join(buf))
            buf = [sent]
            cur = len(sent)
    if buf:
        chunks.append("\n".join(buf))
    return chunks

def detect_context_tag(filename: str) -> str:
    name = filename.lower()
    if "whatsapp" in name:
        return "<context:chat> "
    if "linkedin" in name:
        return "<context:linkedin> "
    if "log" in name:
        return "<context:log> "
    if "essay" in name:
        return "<context:essay> "
    return "<context:generic> "

def load_raw_texts():
    texts = []
    for root, _, files in os.walk(RAW):
        for name in files:
            if not name.lower().endswith(".txt"):
                continue
            p = Path(root) / name
            with open(p, "r", encoding="utf-8", errors="ignore") as file:
                raw = file.read()
            cleaned = clean_text(raw)
            if len(cleaned) < 20:
                continue

            tag = detect_context_tag(name)

            for seg in split_into_utterances(cleaned, max_len=160, group=False):
                if len(seg) >= 20:
                    texts.append(tag + seg)
    
    return texts

def dump_jsonl(texts, path):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for t in texts:
            file.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

def main():
    texts = load_raw_texts()
    if not texts:
        print("No .txt files found")
        return

    random.shuffle(texts)
    n = max(1, int(0.90 * len(texts)))
    
    cleaned_text = [scrub_names(t) for t in texts]
    
    dump_jsonl(cleaned_text[:n], OUT / "mahad_train.jsonl")
    dump_jsonl(cleaned_text[n:], OUT / "mahad_val.jsonl")
    print("Wrote:", OUT / "mahad_train.jsonl", "and", OUT / "mahad_val.jsonl")

if __name__ == "__main__":
    main()