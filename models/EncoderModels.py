import torch
import torch.nn as nn
from torch.nn import functional as F

from modules import EncoderLayer, LearnablePositionEmbedding

class MiniBERT(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, max_length, vocab_size):
        super().__init__()
        self.input_embedding = nn.Embedding(
            vocab_size, embed_size
        )
        
        self.position_embedding = LearnablePositionEmbedding(
            max_length, 
            embed_size
        )
        
        self.encoder = nn.ModuleList(
            [EncoderLayer(embed_size, num_heads) for i in range(num_layers)]
        )
        

    def forward(self, input_seq):
        pe = self.position_embedding(input_seq)
        we = self.input_embedding(input_seq)
        embedding = pe + we
        for layer in self.encoder:
            embedding = layer(embedding)
        
        # return the corresponding output of CLS tokens
        return embedding[:, 0, :]


class TaskMLP(nn.Module):
    def __init__(self, embed_size, num_classes):
        self.net = nn.Sequence(
            nn.Linear(embed_size, 4*embed_size),
            nn.ReLU(),
            nn.Linear(4*embed_size, 4*embed_size),
            nn.ReLU(),
            nn.Linear(4*embed_size, num_classes),
            nn.softmax()
        )
    def forward(self, x):
        return self.net(x)
        