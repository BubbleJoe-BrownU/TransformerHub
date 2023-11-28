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
    
def _make_sliding_window_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: int = 4096,
):
    """
    Make causal mask used for sliding window attention
    """
    bsz, tgt_len = input_ids_shape

    tensor = torch.full(
        (tgt_len, tgt_len),
        fill_value=1,
        device=device,
    )
    mask = torch.tril(tensor, diagonal=0)
    # make the mask banded to account for sliding window
    mask = torch.triu(mask, diagonal=-sliding_window)
    mask = torch.log(mask).to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

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
        self.factor = 1.0 / (10000 ** (torch.repeat_interleave(torch.arange(depth), 2, dim=-1) / embed_size))
        # cos(x) = sin(x + pi/2) so we only use sine function
        self.offset = torch.zeros([embed_size])
        self.offset[1::2] = math.pi/2

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

def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = torch.einsum("i,j->ij", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def duplicaet_interleave(m):
    dim0 = m.shape[0]
    m = m.view(-1, 1)
    m = m.repeat(1, 2)
    m = m.view(dim0, -1)
    return m

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicaet_interleave(t*scale), (sin, cos))
    return (x * cos) + (rotate_every_two(x) * sin)

class XPos(nn.Module):
    """
    Extrapolable Position Embedding
    https://github.com/microsoft/torchscale/blob/881d03079da7b0c52ba0a473c70faac47042efc8/torchscale/component/xpos_relative_position.py#L32
    """
    def __init__(self, ctx_length, embed_size, scale_base=512):
        super().__init__()
        self.embed_size = embed_size
        self.ctx_length = ctx_length
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, embed_size, 2) + 0.4*embed_size) / (1.4*embed_size)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + min_pos + offset
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        
        if downscale:
            scale = 1/scale
        
        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x


