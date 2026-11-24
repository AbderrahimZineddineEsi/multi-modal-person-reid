"""Body ReID embedder: CLIP-ReID ViT-B/16 (own implementation) or OSNet x1.0 (boxmot weights)."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from reid.config import Config
from reid.models.device import resolve_device, use_half, weights_dir

log = logging.getLogger(__name__)

#: Official CLIP-ReID ViT-B/16 MSMT17 checkpoint ("ViT-CLIP-ReID" row of the README table of
#: Syliz517/CLIP-ReID; Google Drive). Override with ``body_reid.weights_url`` (e.g. a mirror).
CLIPREID_MSMT17_URL = "https://drive.google.com/uc?id=1BVaZo93kOksYLjFNH3Gf7JxIbPlWSkcO"
#: boxmot's converted torchreid OSNet x1.0 MSMT17 checkpoint (Google Drive).
OSNET_MSMT17_URL = "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M"

_BACKENDS = {
    # name -> (backend, cache filename, default url, mean, std, embedding dim)
    "clipreid_msmt17": (
        "clipreid",
        "clipreid_vit_b16_msmt17.pth",
        CLIPREID_MSMT17_URL,
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        1280,
    ),
    "osnet_x1_0_msmt17": (
        "osnet",
        "osnet_x1_0_msmt17.pt",
        OSNET_MSMT17_URL,
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
        512,
    ),
}
_IGNORED_PREFIXES = ("classifier.", "classifier_proj.", "prompt_learner.", "text_encoder.")


def download_weights(url: str, dest: Path) -> Path:
    """Fetch ``url`` to ``dest`` unless it exists. Google Drive links/ids go through gdown."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s -> %s", url, dest)
    if "drive.google.com" in url or "://" not in url:
        import gdown

        target = url if "://" in url else f"https://drive.google.com/uc?id={url}"
        if gdown.download(target, str(tmp), quiet=False) is None:
            raise RuntimeError(f"gdown could not download {url} (quota exceeded? set weights_url)")
    else:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (user-configured URL)
    tmp.rename(dest)
    return dest


def preprocess_crops(
    crops: list[np.ndarray], h: int, w: int, mean: tuple[float, ...], std: tuple[float, ...]
) -> torch.Tensor:
    """BGR crops (any size) -> float32 NCHW tensor: stretch-resize to HxW, RGB, normalize.

    Plain stretch without aspect preservation, as in v1 and in the CLIP-ReID training pipeline.
    """
    batch = np.empty((len(crops), h, w, 3), np.float32)
    for i, c in enumerate(crops):
        if c is None or c.size == 0:
            batch[i] = 0.0
            continue
        batch[i] = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    batch /= 255.0
    batch = (batch - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()


class BodyReID:
    """Lazy-loaded body embedder: ``embed(crops) -> (N, D) float32, rows L2-normalized``."""

    def __init__(self, cfg: Config) -> None:
        name = cfg.body_reid.model
        if name not in _BACKENDS:
            raise ValueError(f"unknown body_reid.model {name!r}; choose from {sorted(_BACKENDS)}")
        self.name = name
        self.backend, self._file, url, self.mean, self.std, self.dim = _BACKENDS[name]
        self.url = cfg.body_reid.weights_url or url
        self.h, self.w = cfg.body_reid.input_h, cfg.body_reid.input_w
        self.neck_feat = cfg.body_reid.neck_feat
        self.batch = cfg.runtime.batch_size
        self.device = resolve_device(cfg.runtime.embed_device)
        self.half = use_half(self.device, cfg.runtime.fp16)
        self.weights_dir = weights_dir(cfg.paths.weights_dir)
        self._net: Any = None
        if cfg.body_reid.dim != self.dim:
            raise ValueError(
                f"body_reid.dim={cfg.body_reid.dim} but {name} outputs {self.dim}-D embeddings"
            )

    # ------------------------------------------------------------------ loading
    def load(self) -> BodyReID:
        """Download (if needed) and load the weights (idempotent)."""
        if self._net is not None:
            return self
        path = download_weights(self.url, self.weights_dir / self._file)
        self._net = (
            self._load_clipreid(path) if self.backend == "clipreid" else self._load_osnet(path)
        )
        log.info("body ReID %s (%d-D) on %s fp16=%s", self.name, self.dim, self.device, self.half)
        return self

    def _load_clipreid(self, path: Path) -> torch.nn.Module:
        from reid.models.clipreid_net import ClipReidNet

        raw = torch.load(path, map_location="cpu", weights_only=True)
        dropped = [
            k
            for k in raw
            if not k.removeprefix("module.").startswith(
                ("image_encoder.", "bottleneck.", "bottleneck_proj.")
            )
        ]
        bad = [k for k in dropped if not k.removeprefix("module.").startswith(_IGNORED_PREFIXES)]
        if bad:
            raise RuntimeError(f"unexpected checkpoint keys: {bad[:5]}")
        net = ClipReidNet(self.neck_feat)
        res = net.load_state_dict(ClipReidNet.filter_state_dict(raw), strict=False)
        if res.missing_keys or res.unexpected_keys:
            raise RuntimeError(f"CLIP-ReID weights mismatch: {res}")
        # fp16 runs through torch.autocast in embed(): the weights stay fp32, so the fp32
        # LayerNorms of the CLIP encoder keep fp32 weights (net.half() broke them on a T4).
        return net.eval().to(self.device)

    def _load_osnet(self, path: Path) -> Any:
        from boxmot.reid.core.runtime import ReID

        backend = ReID(path, device=self.device, half=self.half).model
        return backend

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed BGR crops; returns ``(N, D)`` float32 with unit L2 norm (zeros if invalid)."""
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        self.load()
        outs = []
        for s in range(0, len(crops), self.batch):
            x = preprocess_crops(crops[s : s + self.batch], self.h, self.w, self.mean, self.std)
            x = x.to(self.device)
            if self.backend == "clipreid":
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
                    f = self._net(x)
            else:  # boxmot converts OSNet's weights itself when half=True
                f = self._net.forward(x.half() if self.half else x)
            if isinstance(f, (list, tuple)):
                f = f[0]
            f = f.float()
            outs.append(torch.nn.functional.normalize(f, dim=1).cpu())
        emb = torch.cat(outs).numpy().astype(np.float32)
        return np.where(np.isfinite(emb), emb, 0.0)
