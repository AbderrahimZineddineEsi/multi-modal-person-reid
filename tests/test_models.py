"""Model wrappers. Unmarked tests need torch only; ``models`` tests need weights/GPU."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reid.config import Config, load_config  # noqa: E402
from reid.models.clipreid_net import ClipReidNet, _Attention  # noqa: E402


def test_attention_matches_nn_multiheadattention():
    d, heads = 64, 4
    ref = torch.nn.MultiheadAttention(d, heads).eval()  # official CLIP block uses this module
    mine = _Attention(d, heads).eval()
    sd = {k: v for k, v in ref.state_dict().items()}
    assert set(sd) == set(mine.state_dict())  # same parameter names as the official checkpoint
    mine.load_state_dict(sd)
    x = torch.randn(3, 9, d)
    out_ref = ref(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1), need_weights=False)[0]
    assert torch.allclose(mine(x), out_ref.transpose(0, 1), atol=1e-5)


def test_clipreid_net_shapes_and_key_layout():
    net = ClipReidNet("after").eval()
    keys = set(net.state_dict())
    assert "image_encoder.positional_embedding" in keys and "bottleneck_proj.running_var" in keys
    assert "image_encoder.transformer.resblocks.11.attn.in_proj_weight" in keys
    assert net.state_dict()["image_encoder.positional_embedding"].shape == (129, 768)
    with torch.inference_mode():
        out = net(torch.randn(2, 3, 256, 128))
    assert out.shape == (2, 1280)
    net.neck_feat = "before"
    with torch.inference_mode():
        assert net(torch.randn(1, 3, 256, 128)).shape == (1, 1280)
    kept = ClipReidNet.filter_state_dict(
        {
            "module.image_encoder.proj": 1,
            "classifier.weight": 2,
            "text_encoder.x": 3,
            "bottleneck.weight": 4,
        }
    )
    assert set(kept) == {"image_encoder.proj", "bottleneck.weight"}


def test_preprocess_and_dim_mismatch():
    from reid.models.body_reid import BodyReID, preprocess_crops

    x = preprocess_crops(
        [np.full((50, 20, 3), 255, np.uint8), np.zeros((0, 0, 3), np.uint8)],
        256,
        128,
        (0.5,) * 3,
        (0.5,) * 3,
    )
    assert (
        x.shape == (2, 3, 256, 128) and x[0].min() == pytest.approx(1.0) and x[1].max() == -1.0
    )  # empty crop = black image
    with pytest.raises(ValueError):
        BodyReID(Config.model_validate({"body_reid": {"model": "osnet_x1_0_msmt17"}}))  # dim 1280


@pytest.mark.models
def test_clipreid_official_weights_load_strictly():
    from reid.models.body_reid import BodyReID

    b = BodyReID(Config()).load()  # raises on missing/unexpected keys
    e = b.embed([np.random.randint(0, 255, (180, 70, 3), np.uint8) for _ in range(3)])
    assert e.shape == (3, 1280) and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)


@pytest.mark.models
def test_lite_detector_and_osnet_run():
    from reid.models.registry import build_models

    m = build_models(load_config(preset="lite"))
    boxes, confs, kps = m.detector.detect(np.zeros((480, 640, 3), np.uint8))
    assert boxes.shape[1] == 4 and kps is not None
    assert m.body.embed([np.zeros((100, 50, 3), np.uint8)]).shape == (1, 512)


@pytest.mark.models
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("preset", ["balanced", "lite"])  # CLIP-ReID, OSNet
def test_body_reid_fp16_matches_fp32(preset, monkeypatch):
    """fp16 is only enabled on GPUs with fast half (T4 and newer), so force it here.

    Regression: CLIP-ReID converted with ``net.half()`` crashed in its fp32 LayerNorm
    ("expected scalar type Float but found Half") on Colab's T4.
    """
    import reid.models.body_reid as body_reid

    cfg = load_config(None, preset=preset)
    crops = [np.random.default_rng(i).integers(0, 255, (256, 96, 3), np.uint8) for i in range(4)]
    ref = body_reid.BodyReID(cfg).embed(crops)
    monkeypatch.setattr(body_reid, "use_half", lambda device, want: device.startswith("cuda"))
    half = body_reid.BodyReID(cfg)
    assert half.half
    emb = half.embed(crops)
    assert emb.dtype == np.float32 and emb.shape == ref.shape
    assert np.all((emb * ref).sum(1) > 0.99)  # same embeddings up to fp16 rounding
