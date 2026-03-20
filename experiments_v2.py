#!/usr/bin/env python3
"""
Overnight GPU experiment suite v2.
Based on interpretability findings:
- Layer 0 MLP is +171% instrumental, layer 8 is +48%
- Middle layer attention contributes <2% to loss
- Effective rank drops from ~400 to ~250 in late layers
- Prev-token shift freed up 6+ attention heads

Experiments:
1. multi_shift: shifts at distances 1,2,3,4
2. conv: small conv1d (kernel 4) replacing shifts
3. funnel: wide first/last layers, narrow middle
4. mlp_only_mid: remove attention from layers 2-6
5. depth_recur: 4 unique full-rank layers looped 2-3x
6. attn_resid: shared QK, per-layer V (already in experiments.py)
"""
from __future__ import annotations
import math, os, sys, time, pickle, zlib
from pathlib import Path
import numpy as np
import sentencepiece as spm
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from train_gpt_mlx import (
    COMPUTE_DTYPE, CONTROL_TENSOR_NAME_PATTERNS, Hyperparameters,
    TokenLoader, build_sentencepiece_luts, load_validation_tokens,
    eval_val, quantize_state_dict_int8, dequantize_state_dict_int8,
    rms_norm, CastedLinear, RMSNormNoWeight, CausalSelfAttention, MLP, Block,
    Muon, validate_dataset_tokenizer_pair,
)
from experiments import SimpleOptimizers, train_experiment


# ==============================================================================
# 1. MULTI-SHIFT: shifts at distances 1,2,3,4
# ==============================================================================

class MultiShiftBlock(nn.Module):
    """Block with multiple shift distances — gives free n-gram features."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 max_shift=4):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.max_shift = max_shift
        # One weight vector per shift distance
        self.shift_weights = mx.zeros((max_shift, dim), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32))))

    def __call__(self, x, x0):
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        for s in range(self.max_shift):
            pad = mx.zeros_like(x[:, :s+1, :])
            x_shifted = mx.concatenate([pad, x[:, :-(s+1), :]], axis=1)
            x = x + self.shift_weights[s].astype(x.dtype)[None, None, :] * x_shifted
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT_MultiShift(nn.Module):
    def __init__(self, vocab_size, num_layers, dim, num_heads, num_kv_heads, mlp_mult,
                 logit_softcap, rope_base, tied_embed_init_std, qk_gain_init, max_shift=4):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            MultiShiftBlock(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, max_shift)
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std).astype(COMPUTE_DTYPE)

    def __call__(self, input_ids):
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x; skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips: x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def loss(self, input_ids, target_ids):
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self.logit_softcap * mx.tanh((x @ self.tok_emb.weight.astype(x.dtype).T) / self.logit_softcap)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ==============================================================================
# 2. CONV: learned conv1d (kernel 4) replacing shifts
# ==============================================================================

class ConvBlock(nn.Module):
    """Block with a small conv1d for local features instead of shifts."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 conv_kernel=4):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.conv_scale = mx.zeros((1,), dtype=mx.float32)
        # Depthwise conv: each channel independently, causal (pad left)
        self.conv_kernel = conv_kernel
        self.conv_weight = mx.zeros((dim, conv_kernel), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32))))

    def __call__(self, x, x0):
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        # Causal depthwise conv: pad left, no future tokens
        pad = mx.zeros((x.shape[0], self.conv_kernel - 1, x.shape[2]), dtype=x.dtype)
        x_padded = mx.concatenate([pad, x], axis=1)  # (B, T+K-1, d)
        # Manual depthwise conv via sliding window sum
        conv_out = mx.zeros_like(x)
        for k in range(self.conv_kernel):
            conv_out = conv_out + self.conv_weight[:, k].astype(x.dtype)[None, None, :] * x_padded[:, k:k+x.shape[1], :]
        x = x + self.conv_scale.astype(x.dtype) * conv_out
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT_Conv(nn.Module):
    def __init__(self, vocab_size, num_layers, dim, num_heads, num_kv_heads, mlp_mult,
                 logit_softcap, rope_base, tied_embed_init_std, qk_gain_init, conv_kernel=4):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [ConvBlock(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, conv_kernel) for _ in range(num_layers)]
        self.final_norm = RMSNormNoWeight()
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std).astype(COMPUTE_DTYPE)

    def __call__(self, input_ids):
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x; skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips: x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def loss(self, input_ids, target_ids):
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self.logit_softcap * mx.tanh((x @ self.tok_emb.weight.astype(x.dtype).T) / self.logit_softcap)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ==============================================================================
# 3. FUNNEL: wide first/last layers, narrow middle
# ==============================================================================
# Based on instrumentality: layer 0 (+142%) and layer 8 (+48%) are critical.
# Give them 2x MLP width, shrink middle layers.

class GPT_Funnel(nn.Module):
    def __init__(self, vocab_size, num_layers, dim, num_heads, num_kv_heads,
                 logit_softcap, rope_base, tied_embed_init_std, qk_gain_init,
                 edge_mlp_mult=4, mid_mlp_mult=1):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = []
        for i in range(num_layers):
            if i == 0 or i == num_layers - 1:
                mult = edge_mlp_mult
            else:
                mult = mid_mlp_mult
            self.blocks.append(Block(dim, num_heads, num_kv_heads, mult, rope_base, qk_gain_init))
        self.final_norm = RMSNormNoWeight()
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std).astype(COMPUTE_DTYPE)

    def __call__(self, input_ids):
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x; skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips: x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def loss(self, input_ids, target_ids):
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self.logit_softcap * mx.tanh((x @ self.tok_emb.weight.astype(x.dtype).T) / self.logit_softcap)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ==============================================================================
# 4. MLP-ONLY MIDDLE: remove attention from layers 2-6
# ==============================================================================
# Middle layer attention contributes <2% to loss. Replace with just MLP + shift.

class MLPOnlyBlock(nn.Module):
    """Block with no attention — just shift + MLP. For cheap middle layers."""
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.mlp_norm = RMSNormNoWeight()
        self.mlp = MLP(dim, mlp_mult)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.shift_weight = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, x, x0):
        x_shifted = mx.concatenate([mx.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1)
        x = x + self.shift_weight.astype(x.dtype)[None, None, :] * x_shifted
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT_MLPOnlyMid(nn.Module):
    """Layers 0,1 and 7,8 have full attention. Layers 2-6 are MLP-only + shift.
    Saves ~5 layers of attention params (~3.9M) to spend elsewhere."""
    def __init__(self, vocab_size, num_layers, dim, num_heads, num_kv_heads, mlp_mult,
                 logit_softcap, rope_base, tied_embed_init_std, qk_gain_init,
                 mlp_only_start=2, mlp_only_end=7):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = []
        for i in range(num_layers):
            if mlp_only_start <= i < mlp_only_end:
                self.blocks.append(MLPOnlyBlock(dim, mlp_mult))
            else:
                self.blocks.append(Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init))
        self.final_norm = RMSNormNoWeight()
        for b in self.blocks:
            if hasattr(b, 'attn'):
                b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std).astype(COMPUTE_DTYPE)

    def __call__(self, input_ids):
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x; skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips: x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def loss(self, input_ids, target_ids):
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self.logit_softcap * mx.tanh((x @ self.tok_emb.weight.astype(x.dtype).T) / self.logit_softcap)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ==============================================================================
# 5. DEPTH RECURRENCE: full-rank shared layers looped
# ==============================================================================

class GPT_DepthRecur(nn.Module):
    """N unique full-rank layers, looped M times. Same params, more depth.
    No U-Net skips (they don't work well with recurrence)."""
    def __init__(self, vocab_size, num_unique, num_loops, dim, num_heads, num_kv_heads,
                 mlp_mult, logit_softcap, rope_base, tied_embed_init_std, qk_gain_init):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.num_unique = num_unique
        self.num_loops = num_loops
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_unique)
        ]
        self.final_norm = RMSNormNoWeight()
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std).astype(COMPUTE_DTYPE)

    def __call__(self, input_ids, num_loops=None):
        if num_loops is None:
            num_loops = self.num_loops
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        for _ in range(num_loops):
            for block in self.blocks:
                x = block(x, x0)
        return self.final_norm(x)

    def loss(self, input_ids, target_ids):
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self.logit_softcap * mx.tanh((x @ self.tok_emb.weight.astype(x.dtype).T) / self.logit_softcap)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ==============================================================================
# RUNNERS
# ==============================================================================

def run_multi_shift(args):
    model = GPT_MultiShift(
        vocab_size=args.vocab_size, num_layers=args.num_layers, dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std, qk_gain_init=args.qk_gain_init,
    )
    return train_experiment(model, args, f"exp_multi_shift_{args.run_id}", "multi_shift (distances 1-4)")


def run_conv(args, kernel=4):
    model = GPT_Conv(
        vocab_size=args.vocab_size, num_layers=args.num_layers, dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std, qk_gain_init=args.qk_gain_init,
        conv_kernel=kernel,
    )
    return train_experiment(model, args, f"exp_conv_k{kernel}_{args.run_id}", f"conv kernel={kernel}")


def run_funnel(args, edge_mult=4, mid_mult=1):
    model = GPT_Funnel(
        vocab_size=args.vocab_size, num_layers=args.num_layers, dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std, qk_gain_init=args.qk_gain_init,
        edge_mlp_mult=edge_mult, mid_mlp_mult=mid_mult,
    )
    return train_experiment(model, args, f"exp_funnel_e{edge_mult}_m{mid_mult}_{args.run_id}",
                           f"funnel edge_mult={edge_mult} mid_mult={mid_mult}")


def run_mlp_only_mid(args):
    model = GPT_MLPOnlyMid(
        vocab_size=args.vocab_size, num_layers=args.num_layers, dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std, qk_gain_init=args.qk_gain_init,
    )
    return train_experiment(model, args, f"exp_mlp_only_mid_{args.run_id}", "mlp_only_mid (layers 2-6 no attention)")


def run_depth_recur(args, num_unique=4, num_loops=2):
    model = GPT_DepthRecur(
        vocab_size=args.vocab_size, num_unique=num_unique, num_loops=num_loops,
        dim=args.model_dim, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult, logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std, qk_gain_init=args.qk_gain_init,
    )
    return train_experiment(model, args, f"exp_depth_recur_u{num_unique}_l{num_loops}_{args.run_id}",
                           f"depth_recur unique={num_unique} loops={num_loops}")


# ==============================================================================
# 6. STACKED: combine winning techniques
# ==============================================================================

class StackedAttnResidBlock(nn.Module):
    """Shared QK + per-layer V + shift + conv. The kitchen sink."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, shared_qk, conv_kernel=4):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.shared_qk = shared_qk
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        # Shift
        self.shift_weight = mx.zeros((dim,), dtype=mx.float32)
        # Conv
        self.conv_kernel = conv_kernel
        self.conv_weight = mx.zeros((dim, conv_kernel), dtype=mx.float32)
        self.conv_scale = mx.zeros((1,), dtype=mx.float32)

    def __call__(self, x, x0):
        # Shift
        x_shifted = mx.concatenate([mx.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1)
        x = x + self.shift_weight.astype(x.dtype)[None, None, :] * x_shifted
        # Conv
        pad = mx.zeros((x.shape[0], self.conv_kernel - 1, x.shape[2]), dtype=x.dtype)
        x_padded = mx.concatenate([pad, x], axis=1)
        conv_out = mx.zeros_like(x)
        for k in range(self.conv_kernel):
            conv_out = conv_out + self.conv_weight[:, k].astype(x.dtype)[None, None, :] * x_padded[:, k:k+x.shape[1], :]
        x = x + self.conv_scale.astype(x.dtype) * conv_out
        # Attention with shared QK, per-layer V
        x_normed = self.attn_norm(x)
        bsz, seqlen, dim = x_normed.shape
        v = self.c_v(x_normed).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        attn_out = self.shared_qk(x_normed, v)
        attn_out = self.proj(attn_out)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT_Stacked(nn.Module):
    """Everything stacked: shared QK + per-layer V + shift + conv + funnel MLP.
    Edge layers (0, N-1) get wider MLP, middle layers get narrower."""
    def __init__(self, vocab_size, num_layers, dim, num_heads, num_kv_heads,
                 logit_softcap, rope_base, tied_embed_init_std, qk_gain_init,
                 edge_mlp_mult=3, mid_mlp_mult=2, conv_kernel=4):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)

        from experiments import SharedQKAttention
        self.shared_qk = SharedQKAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)

        self.blocks = []
        for i in range(num_layers):
            mult = edge_mlp_mult if (i == 0 or i == num_layers - 1) else mid_mlp_mult
            self.blocks.append(StackedAttnResidBlock(
                dim, num_heads, num_kv_heads, mult, self.shared_qk, conv_kernel,
            ))
        self.final_norm = RMSNormNoWeight()
        for b in self.blocks:
            b.proj.weight = mx.zeros_like(b.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std).astype(COMPUTE_DTYPE)

    def __call__(self, input_ids):
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x; skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips: x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def loss(self, input_ids, target_ids):
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self.logit_softcap * mx.tanh((x @ self.tok_emb.weight.astype(x.dtype).T) / self.logit_softcap)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


def run_stacked(args, edge_mult=3, mid_mult=2, conv_kernel=4):
    model = GPT_Stacked(
        vocab_size=args.vocab_size, num_layers=args.num_layers, dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std, qk_gain_init=args.qk_gain_init,
        edge_mlp_mult=edge_mult, mid_mlp_mult=mid_mult, conv_kernel=conv_kernel,
    )
    return train_experiment(model, args, f"exp_stacked_e{edge_mult}_m{mid_mult}_k{conv_kernel}_{args.run_id}",
                           f"stacked: shared_qk + shift + conv{conv_kernel} + funnel({edge_mult}/{mid_mult})")


EXPERIMENTS_V2 = {
    "multi_shift": run_multi_shift,
    "conv": run_conv,
    "funnel": run_funnel,
    "mlp_only_mid": run_mlp_only_mid,
    "depth_recur": run_depth_recur,
    "stacked": run_stacked,
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=list(EXPERIMENTS_V2.keys()))
    parser.add_argument("--conv-kernel", type=int, default=4)
    parser.add_argument("--edge-mult", type=int, default=4)
    parser.add_argument("--mid-mult", type=int, default=1)
    parser.add_argument("--num-unique", type=int, default=4)
    parser.add_argument("--num-loops", type=int, default=2)
    cli = parser.parse_args()
    args = Hyperparameters()

    if cli.experiment == "conv":
        run_conv(args, kernel=cli.conv_kernel)
    elif cli.experiment == "funnel":
        run_funnel(args, edge_mult=cli.edge_mult, mid_mult=cli.mid_mult)
    elif cli.experiment == "depth_recur":
        run_depth_recur(args, num_unique=cli.num_unique, num_loops=cli.num_loops)
    elif cli.experiment == "stacked":
        run_stacked(args, edge_mult=cli.edge_mult, mid_mult=cli.mid_mult, conv_kernel=cli.conv_kernel)
    else:
        EXPERIMENTS_V2[cli.experiment](args)
