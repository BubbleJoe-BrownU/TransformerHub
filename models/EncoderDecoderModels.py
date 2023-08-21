import torch
import torch.nn as nn
from torch.nn import functional as F

from modules import EncoderLayer,
                    DecoderLayer

class EncoderDecoderModel(nn.Module):
    """An encoder-decoder architecture that was used in the 2017 paper Attention is All You Need for machine translation tasks. Theoretically this architecture can be used in any sequence-to-sequence (Seq2Seq) tasks like text summarization, paraphrasing, image captioning and etc.
    """
    def __init__(self, embed_size, num_head, num_layers, vocab_size):
        super().__init__()
        self.encoder = nn.ModuleList(
            [EncoderLayer(embed_size, num_head) for i in range(num_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(embed_size, num_head) for i in range(num_layers)]
        )
        self.input_embedding = nn.Embedding(
            vocab_size, embed_size
        )
        # using one embedding matrix for input and output is a common practice
        # in language models, called weight tying
        self.lm_head = nn.Linear(embed_size, vocab_size)
        self.lm_head.weight = self.input_embedding.weight

    def forward(self, x):
        context = self.encoder(x)
        
        
        