# Date: August 2023
# Author: Jiayu Zheng
# Email: jiayu_zheng@brown.edu
# Most of implementations were adopted from my final project for Deep Learning CSCI2470, which is written in TensforFlow
# Also influenced by Karpathy's implementation of NanoGPT
# 

import os
import pickle
import requests
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

#
# Single Attention Heads
#

class SelfAttentionHead(nn.Module):
    ''' A self-attention head where the input gather weight sum from itself.
    Can be used on its own or as a building block of multihead attention.

    Keyword arguments:
    embed_size -- the hidden size of embeddings
    head_size -- the size of query, key, and value vectors
    mask -- whether to use causal mask (if to perform next token prediction) or not
    '''
    def __init__(self, embed_size, head_size, mask, ctx_length, dropout):
        super().__init__()
        self.M_key   = nn.Linear(embed_size, head_size, bias=False)
        self.M_query = nn.Linear(embed_size, head_size, bias=False)
        self.M_value = nn.Linear(embed_size, head_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        # masked attention or not
        self.mask = mask
        if self.mask:
            # used to generate the causal mask
            self.register_buffer(name="tril", tensor=torch.tril(torch.ones(ctx_length, ctx_length)).view(1, ctx_length, ctx_length))
        
    def forward(self, x):
        seq_len = x.shape[1]
        ebd_dim = x.shape[-1]
        # get key, query, value embeddings
        k = self.M_key(x)
        q = self.M_query(x)
        v = self.M_value(x)
        # calculate weight and apply causal mask to it
        weight = q @ k.transpose(-2, -1) / ebd_dim**0.5
        if self.mask:
            weight = weight.masked_fill(self.tril[:, :seq_len, :seq_len] == 0, value=float("-inf"))
        weight = F.softmax(weight, dim=-1)
        weight = self.attn_dropout(weight)
        output = weight @ v
        return output

class CrossAttentionHead(nn.Module):
    ''' A cross-attention head where input gather weighted sum from the context

    Keyword arguments:
    embed_size -- the hidden size of embeddings
    head_size -- the size of query, key, and value vectors
    mask -- whether to use causal mask (if to perform next token prediction) or not
    '''
    def __init__(self, embed_size, head_size, dropout):
        super().__init__()
        self.M_key   = nn.Linear(embed_size, head_size, bias=False)
        self.M_query = nn.Linear(embed_size, head_size, bias=False)
        self.M_value = nn.Linear(embed_size, head_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, x, context):
        B, T, C = x.shape
        # get key, query, value embeddings
        q = self.M_query(x)
        k = self.M_key(context)
        v = self.M_value(context)
        # calculate weight and apply causal mask to it
        weight = q @ k.transpose(-2, -1) / C**0.5
        
        weight = F.softmax(weight, dim=-1)
        weight = self.attn_dropout(weight)

        output = weight @ v
        return output

#
# Multi Head Attention Blocks
#

class MultiHeadSelfAttention(nn.Module):
    '''Multihead self-attention
    '''

    def __init__(self, embed_size, num_heads, mask, ctx_length, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_size, embed_size//num_heads, mask, ctx_length, dropout) for i in range(num_heads)
        ])
    
        self.projection = nn.Linear(embed_size//num_heads*num_heads, embed_size)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # concatenate outputs from multiple attention heads
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.resid_dropout(self.projection(output))
        return output

class MultiHeadCrossAttention(nn.Module):
    '''Multihead cross-attention
    '''

    def __init__(self, num_heads, embed_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            CrossAttentionHead(embed_size, embed_size//num_heads, dropout) for i in range(num_heads)
        ])

        self.projection = nn.Linear(embed_size//num_heads*num_heads, embed_size)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, x, context):
        # concatenate outputs from multiple attention heads
        output = torch.cat([h(x, context) for h in self.heads], dim=-1)
        output = self.resid_dropout(self.projection(output))
        return output

class FeedForward(nn.Module):
    '''Feedforward network in Transformer layer
    '''

    def __init__(self, embed_size, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, 4*embed_size),
            nn.GELU(),
            nn.Linear(4*embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)

#
# Transformer Layers
# In practice, a model is made of stacked such layers


class EncoderLayer(nn.Module):
    """A single layer of transformer encoder
    This is typically used in encoder-only and encoder-decoder Transformers

    """
    def __init__(self, embed_size, num_heads, ctx_length, dropout):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_size, num_heads, False, ctx_length, dropout)
        self.ffn = FeedForward(embed_size, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # There is a slight deviation from the original Transformer
        # as a common practice, we move the layernorm before each transformation
        # so-called pre_normalization
        # 
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.ffn(self.layer_norm2(x))
        return x

class DecoderLayer(nn.Module):
    """A single layer of transformer decoder without context or cross-attention.
    This is typically used in a decoder-only Transformer like GPTs

    """
    def __init__(self, embed_size, num_heads, ctx_length, dropout):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_size, num_heads, True, ctx_length, dropout)
        self.ffn = FeedForward(embed_size, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.ffn(self.layer_norm2(x))
        return x

class DecoderLayerWithContext(nn.Module):
    """A single layer of transformer decoder with contex and corss-attention
    This is typically used in a Encoder-Decoder Transformer for Seq2Seq tasks
    """
    def __init__(self, embed_size, num_heads, ctx_length, dropout):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_size, num_heads, True, ctx_length, dropout)
        self.cross_attn = MultiHeadCrossAttention(embed_size, num_heads, dropout)
        self.ffn = FeedForward(embed_size, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.layer_norm3 = nn.LayerNorm(embed_size)

    def forward(self, x, context):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.cross_attn(self.layer_norm2(x), context)
        x = x + self.ffn(self.layer_norm3(x))
        return x

#
# Position Embeddings
# to impose spatial information to embeddings
# the attention operating itself is unordered
#

class PositionEmbedding(nn.Module):
    """The sinusoidal position embedding used in the original Transformer
    """
    def __init__(self, embed_size):
        super().__init__()
        depth = int(embed_size / 2)
        self.factor = torch.cat([torch.arange(0, embed_size, 2).view(-1, 1), torch.arange(0, embed_size, 2).view(-1, 1)], dim=-1).view(-1)
        self.factor = self.factor.to(torch.float16)/embed_size
        self.factor = 1 / 1000 ** self.factor
        # cos(x) = sin(x + pi/2) so we only use sine function
        self.offset = torch.cat([torch.zeros(depth, 1), torch.ones(depth, 1)*math.pi/2], dim=-1).view(-1)

    def forward(self, x):
        # This step should be completed before the function call
        length = x.shape[1]
        pos = torch.arange(length).view(-1, 1)
        position_embedding = torch.sin(pos*self.factor + self.offset)
        return position_embedding
        
        

class LearnablePositionEmbedding(nn.Module):
    """A learnable position embedding

    """
    def __init__(self, ctx_length, embed_size):
        super().__init__()
        self.PE = nn.Embedding(ctx_length, embed_size)
        
    def forward(self, x):
        return self.PE(x)


class LayerNorm(nn.Module):
    """Layer Norm with an optional bias.
    PyTorch doesn't support layer norm with bias=False
    """
    def __init__(self, ndim, bias):
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# class RotaryPositionEmbedding(nn.Module):
#     """A rotary position embedding (RoPE), which encode position information of tokens with a rotation matrix that naturally incorporates explicit relative position dependency.
#     Introduced by Su et. al. in https://arxiv.org/abs/2104.09864

#     Here's a good explanation of the advantages of RoPE, https://nn.labml.ai/transformers/rope/index.html#:~:text=This%20is%20an%20implementation%20of,incorporates%20explicit%20relative%20position%20dependency.
#     """
#     def __init__(self, length: int = 10_000, base: int):
#         self.length = length
#         self.embed_size = embed_size

#         self.cos_cached = None
#         self.sin_cached = None

#     def _build_cache(self, x: torch.Tensor):
#         if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
#             return
#         seq_len = x.shape[0]
#         theta = 1. / (self.length ** (torch.arange(0, self.embed_size, 2).float()/self/embed_size))
#         seq_idx = torch.arange(seq_len).float().to(x.device)
#         idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
#         idx_theta2 = torch.cat([idx_theta, idx_theta]. dim=1)
#         self.cos_cached = idx_theta2.cos()[:, None, None, :]
#         self.sin_cached = idx_theta.sin()[:, None, None, :]
#     def _neg_half(self, x: torch.Tensor):


#     def forward(self, x: torch.Tensor):
#         self._build_cache(x)
#         x_ropem, x_pass = x[..., :self, d]
        
    