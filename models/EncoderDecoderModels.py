import torch
import torch.nn as nn
from torch.nn import functional as F

from modules import EncoderLayer, DecoderLayer, LearnablePositionEmbedding, 

class MiniTransformer(nn.Module):
    """An encoder-decoder architecture that was used in the 2017 paper Attention is All You Need for machine translation tasks. Theoretically this architecture can be used in any sequence-to-sequence (Seq2Seq) tasks like text summarization, paraphrasing, image captioning and etc.
    """
    def __init__(self, embed_size, num_heads, num_layers, ctx_length, vocab_size, dropout, bias):
        super().__init__()
        self.ctx_length = ctx_length
        self.position_embedding = LearnablePositionEmbedding(
            ctx_length,
            embed_size
        )
        self.input_embedding = nn.Embedding(
            vocab_size, embed_size
        )
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [EncoderLayer(embed_size, num_heads, ctx_length, dropout) for i in range(num_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(embed_size, num_heads, ctx_length, dropout) for i in range(num_layers)]
        )
        
        # using one embedding matrix for input and output is a common practice
        # in language models, called weight tying
        self.lm_head = nn.Linear(embed_size, vocab_size)
        self.lm_head.weight = self.input_embedding.weight

    def to_embedding(self, x):
        seq_len = x.shape[1]
        embedding = self.input_embedding(x) + self.position_embedding(torch.arange(seq_len, device=x.device).reshape(-1, 1))
        embedding = self.drop(embedding)
        return embedding
    
    def forward(self, x, context, targets=None):
        """
        x: the beginning of the sequence to be completed
        context: the original paragraph to be summarized
        """
        context = self.to_embedding(context)
        x = self.to_embedding(x)
        for layer in self.encoder:
            context = layer(context)
        for layer in self.decoder:
            x = layer(x, context)
        output = self.layer_norm(x)
        
        if targets is not None:
            logits = self.lm_head(output)
            loss = F.cross_entropy(
                logits.reshape(-1,logits.shape[-1]),
                targets.reshape(-1)
            )
        else:
            logits = self.lm_head(output[:, [-1], :])
            loss = None
        return logits, loss
        
        
        
        
    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        param_dict = {
            pn: p for pn, p in self.named_parameters()
        }
        param_dict = {
            pn: p for pn, p in param_dict.items() if p.requires_grad
        }
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer
        
        
    @torch.no_grad()
    def generate(self, paragraphs, max_len, temperature=1.0, top_k=None):
        """
        Generate the summary of input paragraphs
        """
        
        
        