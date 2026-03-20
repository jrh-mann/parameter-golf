#!/usr/bin/env python3
"""
Overnight experiment runner for parameter-golf.
Architectures: proj-token, routed-layers, lora-routing.
Each experiment runs for a configurable wallclock then evaluates.
"""
from __future__ import annotations

import glob
import math
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# Reuse data loading, quantization, eval from the main script.
from train_gpt_mlx import (
    COMPUTE_DTYPE,
    CONTROL_TENSOR_NAME_PATTERNS,
    Hyperparameters,
    TokenLoader,
    build_sentencepiece_luts,
    load_validation_tokens,
    eval_val,
    quantize_state_dict_int8,
    dequantize_state_dict_int8,
    rms_norm,
    zeropower_newtonschulz5,
    accumulate_flat_grads,
    token_chunks,
    validate_dataset_tokenizer_pair,
    CastedLinear,
    RMSNormNoWeight,
    CausalSelfAttention,
    MLP,
    Block,
    Muon,
)

import pickle
import zlib


# ==============================================================================
# EXPERIMENT 1: PROJ-TOKEN (single-pass, neuralese = proj(embed(t)))
# ==============================================================================

class GPT_ProjToken(nn.Module):
    """Single-pass neuralese: each token gets a thinking slot initialized as proj(embed(t)).
    Sequence: [embed(t0), proj(embed(t0)), embed(t1), proj(embed(t1)), ...]
    One forward pass at 2T. Loss at odd positions only.
    """
    def __init__(self, vocab_size, num_layers, dim, num_heads, num_kv_heads, mlp_mult,
                 logit_softcap, rope_base, tied_embed_init_std, qk_gain_init):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.neuralese_proj = CastedLinear(dim, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()

        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)
        self.neuralese_proj.weight = mx.zeros_like(self.neuralese_proj.weight)

    def __call__(self, input_ids):
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        # Thinking seed: projected embedding per token.
        thinking = self.neuralese_proj(x)
        B, T, d = x.shape
        # Interleave: [real_0, think_0, real_1, think_1, ...]
        x = mx.stack([x, thinking.astype(x.dtype)], axis=2).reshape(B, 2 * T, d)

        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def loss(self, input_ids, target_ids):
        x = self(input_ids)
        # Extract thinking positions (odd indices) for prediction.
        x = x[:, 1::2, :]
        x = x.reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self.logit_softcap * mx.tanh(
            (x @ self.tok_emb.weight.astype(x.dtype).T) / self.logit_softcap
        )
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ==============================================================================
# EXPERIMENT 2: ROUTED SHARED LAYERS
# ==============================================================================

class Router(nn.Module):
    """Tiny router: mean-pool hidden states → pick which layer to run."""
    def __init__(self, dim, num_experts):
        super().__init__()
        self.linear = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts

    def __call__(self, x):
        # (B, T, d) → mean pool → (B, d) → logits (B, num_experts)
        pooled = mx.mean(x, axis=1)
        return self.linear(pooled)


class GPT_Routed(nn.Module):
    """Stem layers + shared expert layers with a learned router.
    The router picks which expert layer to run at each recurrence step.
    Uses Gumbel-softmax for differentiable routing during training.
    """
    def __init__(self, vocab_size, num_stem, num_experts, num_recurrence, dim,
                 num_heads, num_kv_heads, mlp_mult, logit_softcap, rope_base,
                 tied_embed_init_std, qk_gain_init):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.num_stem = num_stem
        self.num_experts = num_experts
        self.num_recurrence = num_recurrence
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.stem = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_stem)
        ]
        self.experts = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_experts)
        ]
        self.router = Router(dim, num_experts)
        self.final_norm = RMSNormNoWeight()

        for b in self.stem + self.experts:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def __call__(self, input_ids, num_recurrence=None):
        if num_recurrence is None:
            num_recurrence = self.num_recurrence
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        # Stem layers always run.
        for block in self.stem:
            x = block(x, x0)
        # Routed recurrence: soft mixture of expert outputs.
        for _ in range(num_recurrence):
            logits = self.router(x)  # (B, num_experts)
            weights = mx.softmax(logits, axis=-1)  # (B, num_experts)
            # Weighted combination of all expert outputs.
            expert_out = mx.zeros_like(x)
            for j, expert in enumerate(self.experts):
                out = expert(x, x0)
                expert_out = expert_out + weights[:, j:j+1, None] * out  # broadcast (B,1,1) * (B,T,d)
            x = expert_out
        return self.final_norm(x)

    def loss(self, input_ids, target_ids):
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self.logit_softcap * mx.tanh(
            (x @ self.tok_emb.weight.astype(x.dtype).T) / self.logit_softcap
        )
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ==============================================================================
# EXPERIMENT 3: LORA ROUTING
# ==============================================================================

class LoRARoutedLinear(nn.Module):
    """A linear layer composed of independently-routed A and B low-rank factors.
    Pool of K A-matrices (out_dim, r) and L B-matrices (r, in_dim).
    Router picks one A and one B. Effective weight = A_i @ B_j.
    K*L possible layers from K+L stored matrices.
    """
    def __init__(self, in_dim, out_dim, rank, num_a, num_b):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.num_a = num_a
        self.num_b = num_b
        scale = 1.0 / math.sqrt(in_dim)
        self.a_pool = mx.random.normal((num_a, out_dim, rank)) * scale
        self.b_pool = mx.zeros((num_b, rank, in_dim))

    def __call__(self, x, a_weights, b_weights):
        A = mx.sum(a_weights[:, None, None] * self.a_pool, axis=0)  # (out_dim, rank)
        B = mx.sum(b_weights[:, None, None] * self.b_pool, axis=0)  # (rank, in_dim)
        # x @ (A @ B)^T = x @ B^T @ A^T
        return (x @ B.T) @ A.T


class LoRARoutedBlock(nn.Module):
    """Transformer block where Q/K/V/O projections and MLP are LoRA-routed."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rank, num_a, num_b,
                 rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        # Standard attention components that aren't routed.
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        # LoRA-routed projections for attention.
        self.c_q = LoRARoutedLinear(dim, dim, rank, num_a, num_b)
        self.c_k = LoRARoutedLinear(dim, kv_dim, rank, num_a, num_b)
        self.c_v = LoRARoutedLinear(dim, kv_dim, rank, num_a, num_b)
        self.proj = LoRARoutedLinear(dim, dim, rank, num_a, num_b)
        # LoRA-routed MLP.
        hidden = dim * mlp_mult
        self.fc = LoRARoutedLinear(dim, hidden, rank, num_a, num_b)
        self.fc_proj = LoRARoutedLinear(hidden, dim, rank, num_a, num_b)
        # Scales.
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x, x0, a_weights, b_weights):
        bsz, seqlen, dim = x.shape
        # Attention.
        x_normed = self.attn_norm(x)
        q = self.c_q(x_normed, a_weights, b_weights).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x_normed, a_weights, b_weights).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x_normed, a_weights, b_weights).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        attn_out = self.proj(y, a_weights, b_weights)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        # MLP with relu^2.
        x_normed = self.mlp_norm(x)
        h = nn.relu(self.fc(x_normed, a_weights, b_weights))
        mlp_out = self.fc_proj(h * h, a_weights, b_weights)
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * mlp_out
        return x


class GPT_LoRARouted(nn.Module):
    """Transformer with LoRA-routed shared layers.
    Stem layers are normal. Recurrent layers use LoRA routing:
    a pool of A and B matrices, router picks which to combine.
    """
    def __init__(self, vocab_size, num_stem, num_recurrence, dim, num_heads, num_kv_heads,
                 mlp_mult, rank, num_a, num_b, logit_softcap, rope_base,
                 tied_embed_init_std, qk_gain_init):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.num_recurrence = num_recurrence
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.stem = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_stem)
        ]
        self.routed_block = LoRARoutedBlock(
            dim, num_heads, num_kv_heads, mlp_mult, rank, num_a, num_b,
            rope_base, qk_gain_init,
        )
        self.a_router = nn.Linear(dim, num_a, bias=False)
        self.b_router = nn.Linear(dim, num_b, bias=False)
        self.final_norm = RMSNormNoWeight()

        for b in self.stem:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def __call__(self, input_ids, num_recurrence=None):
        if num_recurrence is None:
            num_recurrence = self.num_recurrence
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        for block in self.stem:
            x = block(x, x0)
        for _ in range(num_recurrence):
            pooled = mx.mean(x, axis=1)  # (B, d)
            a_weights = mx.softmax(self.a_router(pooled), axis=-1)  # (B, num_a)
            b_weights = mx.softmax(self.b_router(pooled), axis=-1)  # (B, num_b)
            # For LoRARoutedLinear we need (num_a,) weights — take mean over batch for simplicity.
            a_w = mx.mean(a_weights, axis=0)  # (num_a,)
            b_w = mx.mean(b_weights, axis=0)  # (num_b,)
            x = self.routed_block(x, x0, a_w, b_w)
        return self.final_norm(x)

    def loss(self, input_ids, target_ids):
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self.logit_softcap * mx.tanh(
            (x @ self.tok_emb.weight.astype(x.dtype).T) / self.logit_softcap
        )
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ==============================================================================
# EXPERIMENT 4: SPARSE PER-TOKEN MoE (shared attention, routed MLP)
# ==============================================================================

class SparseMLP(nn.Module):
    """MLP with per-token LoRA routing. Each token independently picks A_i and B_j.
    Hard top-1 selection with straight-through gradients.
    """
    def __init__(self, dim, mlp_mult, rank, num_a, num_b):
        super().__init__()
        hidden = dim * mlp_mult
        self.dim = dim
        self.hidden = hidden
        self.rank = rank
        self.num_a = num_a
        self.num_b = num_b
        # Pool for fc (up-projection): x @ B^T @ A^T
        scale_up = 1.0 / math.sqrt(dim)
        self.fc_a_pool = mx.random.normal((num_a, hidden, rank)) * scale_up
        self.fc_b_pool = mx.zeros((num_b, rank, dim))
        # Pool for proj (down-projection): h @ B^T @ A^T
        scale_down = 1.0 / math.sqrt(hidden)
        self.proj_a_pool = mx.random.normal((num_a, dim, rank)) * scale_down
        self.proj_b_pool = mx.zeros((num_b, rank, hidden))
        # Per-token routers.
        self.a_router = nn.Linear(dim, num_a, bias=False)
        self.b_router = nn.Linear(dim, num_b, bias=False)

    def _hard_topk(self, logits):
        """Hard top-1 with straight-through: forward uses argmax one-hot,
        backward pretends it was softmax."""
        soft = mx.softmax(logits, axis=-1)
        idx = mx.argmax(logits, axis=-1)  # (...,)
        hard = mx.zeros_like(soft)
        # One-hot via scatter — use fancy indexing.
        hard = mx.zeros_like(soft)
        # Build one-hot manually.
        one_hot = mx.equal(mx.arange(soft.shape[-1])[None, :], idx[:, None]).astype(soft.dtype)
        # Straight-through: forward = one_hot, backward = soft.
        return mx.stop_gradient(one_hot - soft) + soft

    def _routed_matmul(self, x, a_pool, b_pool, a_hard, b_hard):
        """Per-token routed low-rank matmul. Memory-safe: loops over experts.
        x: (N, in_dim), a_hard/b_hard: (N, num_a/b) one-hot-ish
        a_pool: (num_a, out_dim, rank), b_pool: (num_b, rank, in_dim)
        Since A and B are independently routed, we factor into two small loops.
        """
        num_b = b_pool.shape[0]
        num_a = a_pool.shape[0]
        # Step 1: x @ B^T, weighted by b_hard. Loop over B experts.
        rank = b_pool.shape[1]
        mid = mx.zeros((x.shape[0], rank), dtype=x.dtype)
        for j in range(num_b):
            mid_j = x @ b_pool[j].astype(x.dtype).T  # (N, rank)
            mid = mid + b_hard[:, j:j+1] * mid_j
        # Step 2: mid @ A^T, weighted by a_hard. Loop over A experts.
        out_dim = a_pool.shape[1]
        out = mx.zeros((x.shape[0], out_dim), dtype=x.dtype)
        for i in range(num_a):
            out_i = mid @ a_pool[i].astype(x.dtype).T  # (N, out_dim)
            out = out + a_hard[:, i:i+1] * out_i
        return out

    def __call__(self, x):
        B, T, d = x.shape
        x_flat = x.reshape(-1, d)  # (N, dim)

        # Per-token routing.
        a_logits = self.a_router(x_flat)  # (N, num_a)
        b_logits = self.b_router(x_flat)  # (N, num_b)
        a_hard = self._hard_topk(a_logits)  # (N, num_a) ~one-hot
        b_hard = self._hard_topk(b_logits)  # (N, num_b) ~one-hot

        # Up-projection with relu^2.
        h = self._routed_matmul(x_flat, self.fc_a_pool, self.fc_b_pool, a_hard, b_hard)
        h = nn.relu(h)
        h = h * h
        # Down-projection (reuse same routing decision).
        out = self._routed_matmul(h, self.proj_a_pool, self.proj_b_pool, a_hard, b_hard)
        return out.reshape(B, T, d)


class SparseBlock(nn.Module):
    """Transformer block: shared full-rank attention + per-token routed sparse MLP."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rank, num_a, num_b,
                 rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        # Shared full-rank attention.
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        # Per-token routed MLP.
        self.mlp = SparseMLP(dim, mlp_mult, rank, num_a, num_b)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x, x0):
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        mlp_out = self.mlp(self.mlp_norm(x))
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * mlp_out
        return x


class GPT_SparseMoE(nn.Module):
    """Stem layers (full-rank) + recurrent sparse blocks (shared attn, per-token routed MLP).
    Each token independently picks LoRA A_i and B_j for its MLP at each recurrence step.
    """
    def __init__(self, vocab_size, num_stem, num_recurrence, dim, num_heads, num_kv_heads,
                 mlp_mult, rank, num_a, num_b, logit_softcap, rope_base,
                 tied_embed_init_std, qk_gain_init):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.num_recurrence = num_recurrence
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.stem = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_stem)
        ]
        self.sparse_block = SparseBlock(
            dim, num_heads, num_kv_heads, mlp_mult, rank, num_a, num_b,
            rope_base, qk_gain_init,
        )
        self.final_norm = RMSNormNoWeight()

        for b in self.stem:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        # Zero-init the shared attention output proj too.
        self.sparse_block.attn.proj.weight = mx.zeros_like(self.sparse_block.attn.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def __call__(self, input_ids, num_recurrence=None):
        if num_recurrence is None:
            num_recurrence = self.num_recurrence
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        for block in self.stem:
            x = block(x, x0)
        for _ in range(num_recurrence):
            x = self.sparse_block(x, x0)
        return self.final_norm(x)

    def loss(self, input_ids, target_ids):
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self.logit_softcap * mx.tanh(
            (x @ self.tok_emb.weight.astype(x.dtype).T) / self.logit_softcap
        )
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ==============================================================================
# EXPERIMENT 5: ATTENTION RESIDUAL (shared QK, per-layer V)
# ==============================================================================

class SharedQKAttention(nn.Module):
    """Shared Q/K projections + RoPE across all layers. Cheap positional attention."""
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x, v):
        """x provides Q and K (shared), v is pre-computed per-layer values."""
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        return y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)


class AttnResidBlock(nn.Module):
    """Block with attention-based residual: shared QK decides WHERE to look,
    per-layer V decides WHAT to extract, per-layer O projects back."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 shared_qk):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.shared_qk = shared_qk  # shared across layers
        kv_dim = num_kv_heads * (dim // num_heads)
        self.c_v = CastedLinear(dim, kv_dim)  # per-layer V
        self.proj = CastedLinear(dim, dim)     # per-layer O
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.shift_weight = mx.zeros((dim,), dtype=mx.float32)
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads

    def __call__(self, x, x0):
        # Shift
        x_shifted = mx.concatenate([mx.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1)
        x = x + self.shift_weight.astype(x.dtype)[None, None, :] * x_shifted
        # Attention with shared QK, per-layer V
        x_normed = self.attn_norm(x)
        bsz, seqlen, dim = x_normed.shape
        v = self.c_v(x_normed).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        attn_out = self.shared_qk(x_normed, v)
        attn_out = self.proj(attn_out)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT_AttnResid(nn.Module):
    """GPT with shared QK attention across all layers. Each layer has its own V and O.
    Saves ~60% of attention params (Q+K shared, only V+O per layer)."""
    def __init__(self, vocab_size, num_layers, dim, num_heads, num_kv_heads, mlp_mult,
                 logit_softcap, rope_base, tied_embed_init_std, qk_gain_init):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.shared_qk = SharedQKAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            AttnResidBlock(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                          self.shared_qk)
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()

        for b in self.blocks:
            b.proj.weight = mx.zeros_like(b.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def __call__(self, input_ids):
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def loss(self, input_ids, target_ids):
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self.logit_softcap * mx.tanh(
            (x @ self.tok_emb.weight.astype(x.dtype).T) / self.logit_softcap
        )
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ==============================================================================
# OPTIMIZER SETUP (reused across experiments)
# ==============================================================================

class SimpleOptimizers:
    """Muon for 2D matrices, Adam for everything else. Simpler than SplitOptimizers."""
    def __init__(self, model, args):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k for k, p in params.items()
            if p.ndim == 2 and k != self.embed_key
            and not any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k for k, p in params.items()
            if k not in self.matrix_keys and k != self.embed_key
        ]
        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(
            learning_rate=args.tied_embed_lr, betas=[args.beta1, args.beta2],
            eps=args.adam_eps, bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=args.scalar_lr, betas=[args.beta1, args.beta2],
            eps=args.adam_eps, bias_correction=True,
        )

    def step(self, model, grads_tree, step, lr_mul):
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)
        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))
        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        if self.embed_key in grads:
            updated.update(self.adam_embed.apply_gradients(
                {self.embed_key: grads[self.embed_key]},
                {self.embed_key: params[self.embed_key]},
            ))
        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        sg = {k: grads[k] for k in self.scalar_keys if k in grads}
        sp = {k: params[k] for k in self.scalar_keys if k in grads}
        if sg:
            updated.update(self.adam_scalar.apply_gradients(sg, sp))
        model.update(tree_unflatten(list(updated.items())))


# ==============================================================================
# GENERIC TRAINING LOOP
# ==============================================================================

def train_experiment(model, args, run_id, log_prefix=""):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{run_id}.txt"

    def log(msg, console=True):
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path, args.tokenizer_path,
    )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, args.val_max_tokens)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size,
    )

    mx.random.seed(args.seed)
    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)
    opt = SimpleOptimizers(model, args)

    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state, outputs=model.state,
    )

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{run_id}")
    log(f"experiment:{log_prefix}")
    log(f"model_params:{n_params}")
    log(f"seq_len:{args.train_seq_len} train_batch_tokens:{args.train_batch_tokens}")
    log(f"val_tokens:{val_tokens.size - 1}")

    # Warmup.
    for warmup_step in range(args.warmup_steps):
        x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        mx.eval(loss, grads)
        mx.synchronize()
        if warmup_step + 1 == args.warmup_steps:
            log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step = None
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step:
            val_loss, val_bpb = eval_val(
                args, compiled_loss, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{train_time_ms:.0f}ms")
            if stop_after_step is not None:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()
        x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        train_loss_value = float(loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log(f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}")
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # Quantize and roundtrip eval.
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    quant_obj, quant_stats = quantize_state_dict_int8(flat_state)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_file_bytes = len(quant_blob)
    log(f"serialized_model_int8_zlib:{quant_file_bytes} bytes")

    quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob)))
    model.update(tree_unflatten(list(quant_flat.items())))
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_loss, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    log(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")
    log(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    return q_val_bpb


# ==============================================================================
# EXPERIMENT DEFINITIONS
# ==============================================================================

def run_proj_token(args):
    model = GPT_ProjToken(
        vocab_size=args.vocab_size, num_layers=args.num_layers, dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std, qk_gain_init=args.qk_gain_init,
    )
    return train_experiment(model, args, f"proj_token_{args.run_id}", "proj_token")


def run_routed(args, num_stem=3, num_experts=4, num_recurrence=6):
    model = GPT_Routed(
        vocab_size=args.vocab_size, num_stem=num_stem, num_experts=num_experts,
        num_recurrence=num_recurrence, dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std, qk_gain_init=args.qk_gain_init,
    )
    return train_experiment(model, args, f"routed_s{num_stem}_e{num_experts}_r{num_recurrence}_{args.run_id}",
                           f"routed stem={num_stem} experts={num_experts} recurrence={num_recurrence}")


def run_lora_routed(args, num_stem=3, num_recurrence=6, rank=64, num_a=8, num_b=8):
    model = GPT_LoRARouted(
        vocab_size=args.vocab_size, num_stem=num_stem, num_recurrence=num_recurrence,
        dim=args.model_dim, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult, rank=rank, num_a=num_a, num_b=num_b,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std, qk_gain_init=args.qk_gain_init,
    )
    return train_experiment(model, args, f"lora_routed_s{num_stem}_r{num_recurrence}_rank{rank}_a{num_a}_b{num_b}_{args.run_id}",
                           f"lora_routed stem={num_stem} recurrence={num_recurrence} rank={rank} a={num_a} b={num_b}")


# ==============================================================================
# CLI
# ==============================================================================

def run_attn_resid(args):
    model = GPT_AttnResid(
        vocab_size=args.vocab_size, num_layers=args.num_layers, dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std, qk_gain_init=args.qk_gain_init,
    )
    return train_experiment(model, args, f"attn_resid_{args.run_id}", "attn_resid (shared QK, per-layer V)")


def run_sparse_moe(args, num_stem=3, num_recurrence=6, rank=64, num_a=8, num_b=8):
    model = GPT_SparseMoE(
        vocab_size=args.vocab_size, num_stem=num_stem, num_recurrence=num_recurrence,
        dim=args.model_dim, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult, rank=rank, num_a=num_a, num_b=num_b,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std, qk_gain_init=args.qk_gain_init,
    )
    return train_experiment(model, args, f"sparse_moe_s{num_stem}_r{num_recurrence}_rank{rank}_a{num_a}_b{num_b}_{args.run_id}",
                           f"sparse_moe stem={num_stem} recurrence={num_recurrence} rank={rank} a={num_a} b={num_b}")


EXPERIMENTS = {
    "proj_token": run_proj_token,
    "routed": run_routed,
    "lora_routed": run_lora_routed,
    "sparse_moe": run_sparse_moe,
    "attn_resid": run_attn_resid,
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS.keys())
    # LoRA routing knobs.
    parser.add_argument("--num-stem", type=int, default=3)
    parser.add_argument("--num-recurrence", type=int, default=6)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--num-a", type=int, default=8)
    parser.add_argument("--num-b", type=int, default=8)
    # Routed knobs.
    parser.add_argument("--num-experts", type=int, default=4)
    cli = parser.parse_args()

    args = Hyperparameters()
    if cli.experiment in ("lora_routed", "sparse_moe"):
        EXPERIMENTS[cli.experiment](args, num_stem=cli.num_stem, num_recurrence=cli.num_recurrence,
                                   rank=cli.rank, num_a=cli.num_a, num_b=cli.num_b)
    elif cli.experiment == "routed":
        run_routed(args, num_stem=cli.num_stem, num_experts=cli.num_experts,
                  num_recurrence=cli.num_recurrence)
    else:
        EXPERIMENTS[cli.experiment](args)
