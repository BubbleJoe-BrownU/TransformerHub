import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
from dataclasses import dataclass
from modules import DecoderLayer, LearnablePositionEmbedding

@dataclass
class GPTConfig:
    ctx_length: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size is 50257, padded up to nearest multiple of 64 for efficiency
    num_layers: int = 12
    num_heads: int = 12
    embed_size: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorm, like GPT-2. False: a bit better and faster


class MiniGPT(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, ctx_length, vocab_size, dropout, bias):
        super().__init__()
        self.ctx_length = ctx_length
        self.input_embedding = nn.Embedding(
            vocab_size, embed_size
        )
        
        self.position_embedding = LearnablePositionEmbedding(
            ctx_length, 
            embed_size
        )
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.ModuleList(
            [DecoderLayer(embed_size, num_heads, ctx_length, dropout) for i in range(num_layers)]
        )

        self.layer_norm = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)
        # weight tying
        self.lm_head.weight = self.input_embedding.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal(p, mean=0.0, std=0.02/math.sqrt(2*num_layers))

    def forward(self, x, targets=None):
        length = x.shape[1]
        pe = self.position_embedding(torch.arange(length, device=x.device).reshape(1, -1))
        we = self.input_embedding(x)
        embedding = self.drop(pe + we)
        for layer in self.decoder:
            embedding = layer(embedding)
        embedding = self.layer_norm(embedding)
        
        if targets is not None:
            logits = self.lm_head(embedding)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        else:
            logits = self.lm_head(embedding[:, [-1], :])
            loss = None
        return logits, loss
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        
        """
        num_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            num_params -= self.input_embedding.weight.numel()
        return num_params

    
    def crop_model_size(self, ctx_length):
        # model surgery to decrease the context length of the model
        # crop unnecessary parts of weights pertaining to context length
        assert ctx_length < self.ctx_length
        self.ctx_length = ctx_length
        self.position_embedding.PE.weight = nn.Parameter(self.position_embedding.PE.weight[:ctx_length])
        for layer in self.decoder:
            for head in layer.self_attn.heads:
                if hasattr(head, 'tril'):
                    head.tril = head.tril[:, :ctx_length, :ctx_length]
        
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        estimate model flops utilization (MFU) in units of A100 bfloats peak
        """
        # estimate the number of flops we do per iteration
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        L, H, Q, T = self.num_layers, self.num_heads, self.embed_size//self.num_heads, self.ctx_length
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu
    
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
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        # create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_rgs or {}
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {mode_type}")
        
        model_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embed=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embed=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
        

        
    @torch.no_grad()
    def generate(self, seq, max_new_tokens, temperature=1.0, top_k=None):
        """
        generate a sequence of max_new_tokens given an initial sequence
        """
        if max_new_tokens + seq.shape[1] > self.ctx_length:
            print(f"Length of the sequence to be generated : {max_new_tokens + seq.shape[1]} is larger than the max context length {self.ctx_length}, result might be inconsistent!")
        
        for i in range(max_new_tokens):
            # trim the sequence input if the length is greater then ctx_length
            seq_input = seq if seq.size(1) <= self.ctx_length else seq[:, -self.ctx_length:]
            logits, _ = self(seq_input)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                # topk returns the top k largest values and their indices
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_tokens = torch.multinomial(probs, num_samples=1)
            seq = torch.cat((seq, next_tokens), dim=-1)
        return seq