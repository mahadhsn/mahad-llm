import torch
import torch.nn as nn

class MiniLM(nn.Module):
    
    def __init__(self, vocab_size, d_model=256, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 1) token + position embeddigns
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # 2) simple final norm + linear head to vocab
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight,   mean=0.0, std=0.02)
        
    def forward(self, input_ids):
        """
        input_ids: (B, T) int64
        returns logits: (B, T, vocab_size)
        """
        
        B, T = input_ids.shape
        
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} > max_seq_len {self.max_seq_len}")
        
        # positions: 0..T-1
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0) # (1, T)
        
        # embeddings
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        
        # project to vocab 
        x = self.ln_f(x)
        logits = self.head(x)
        return logits