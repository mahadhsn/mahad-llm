# since we are using 0...255, special tokens will be after the 255 mark 

BOS = 256 # Beginning of Sequence
EOS = 257 # End of Sequence
PAD = 258 # Padding

SPECIALS = {
    BOS: "<|bos|>", 
    EOS: "<|eos|>", 
    PAD: "<|pad|>"
}

class ByteLevelTokenizer:
    """
    Minimal tokenizer:
    - encode: text -> list[int] of UTF-8 bytes (+ optional BOS/EOS)
    - decode: list[int] -> text (ignores specials)
    """
    
    def __init__(self, add_bos= False, add_eos= False):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.vocab_size = 259 # 0..255 + 3 specials
        
    def encode(self, text, add_special=None): # encode: text -> list of ints (tokens)
        if add_special is None:
            add_special = self.add_bos or self.add_eos
            
        ids = list(text.encode("utf-8", errors="replace"))
        
        if add_special:
            if self.add_bos:
                ids = [BOS] + ids
            if self.add_eos:
                ids = ids + [EOS]
        return ids
    
    def decode(self, ids): # decode: list of ints (tokens) -> text
        byte_vals = []
        
        for i in ids:
            if 0 <= i <= 255: # remove any special chars
                byte_vals.append(i)
        
        b = bytes(byte_vals) # convert to bytes
        
        return b.decode("utf-8", errors="replace")
        