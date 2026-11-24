"""Inference-only CLIP-ReID ViT-B/16 network (image encoder + BN necks).

Re-implementation of the vision path of the official repository
(Syliz517/CLIP-ReID, ``model/clip/model.py::VisionTransformer`` and
``model/make_model_clipreid.py::build_transformer``, MIT licence, Li et al. 2023,
"CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete
Text Labels"). Parameter names are identical to the official checkpoint so that its
``state_dict`` loads without renaming; the text encoder, prompt learner and the two
classifiers (training-only) are not instantiated and are filtered out when loading.

Differences from the official code: no CUDA assumption, no fp16 weight conversion, no SIE
(camera/view embeddings; the released ViT-CLIP-ReID checkpoints are trained without it) and
``nn.MultiheadAttention`` is replaced by an equivalent ``F.scaled_dot_product_attention`` call
that reads the very same ``attn.in_proj_*`` / ``attn.out_proj.*`` parameters.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class _LayerNorm(nn.LayerNorm):
    """LayerNorm computed in fp32 (CLIP convention, safe under fp16)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class _Attention(nn.Module):
    """Same parameters as ``nn.MultiheadAttention`` (``in_proj_weight`` etc.), batch-first."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d))
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(b, n, d))


class _Block(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.attn = _Attention(d, heads)
        self.ln_1 = _LayerNorm(d)
        self.mlp = nn.Sequential()
        self.mlp.add_module("c_fc", nn.Linear(d, 4 * d))
        self.mlp.add_module("gelu", _QuickGELU())
        self.mlp.add_module("c_proj", nn.Linear(4 * d, d))
        self.ln_2 = _LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class _QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class _Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.resblocks = nn.Sequential(*[_Block(width, heads) for _ in range(layers)])


class ClipReidVisual(nn.Module):
    """CLIP ViT-B/16 image tower with the CLIP-ReID modifications (stride, 16x8 grid)."""

    def __init__(
        self,
        h_res: int = 16,
        w_res: int = 8,
        patch: int = 16,
        stride: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        out_dim: int = 512,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, patch, stride, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.zeros(h_res * w_res + 1, width))
        self.ln_pre = _LayerNorm(width)
        self.transformer = _Transformer(width, layers, heads)
        self.ln_post = _LayerNorm(width)
        self.proj = nn.Parameter(torch.zeros(width, out_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cls tokens (after ``ln_post``, 768-D) and their projection (512-D).

        Mirrors the official forward: blocks 1-12, ``ln_post`` on the last block output, then
        ``@ proj`` (the official code also returns the block-11 tokens, unused at test time).
        """
        x = self.conv1(x).flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.transformer.resblocks(self.ln_pre(x))
        cls_tok = self.ln_post(x[:, 0])
        return cls_tok, cls_tok @ self.proj.to(cls_tok.dtype)


class ClipReidNet(nn.Module):
    """Image encoder + the two BatchNorm necks; ``forward`` is the official eval path."""

    def __init__(self, neck_feat: Literal["before", "after"] = "after") -> None:
        super().__init__()
        self.neck_feat = neck_feat
        self.image_encoder = ClipReidVisual()
        self.bottleneck = nn.BatchNorm1d(768)
        self.bottleneck_proj = nn.BatchNorm1d(512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, feat_proj = self.image_encoder(x)
        if self.neck_feat == "after":
            return torch.cat([self.bottleneck(feat), self.bottleneck_proj(feat_proj)], dim=1)
        return torch.cat([feat, feat_proj], dim=1)

    @staticmethod
    def filter_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Drop training-only keys (classifiers, prompt learner, text tower); strip ``module.``."""
        keep = ("image_encoder.", "bottleneck.", "bottleneck_proj.")
        return {
            k.removeprefix("module."): v
            for k, v in sd.items()
            if k.removeprefix("module.").startswith(keep)
        }
