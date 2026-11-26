"""Attribute head: NumPy inference equals the torch model; head.npz round trip."""

import numpy as np
import pytest

from reid.attributes import head as hd
from reid.attributes.schema import GROUPS, LABELS

torch = pytest.importorskip("torch")
from reid.attributes.mlp import MLPHead  # noqa: E402


def _trained_like_head(in_dim=24, hidden=16):
    torch.manual_seed(0)
    model = MLPHead(in_dim, len(LABELS), hidden, 0.3)
    model.train()
    for _ in range(5):  # move the BatchNorm running statistics away from 0 / 1
        model(torch.randn(64, in_dim) * 2 + 1)
    model.eval()
    with torch.no_grad():
        model.bn.weight.uniform_(0.5, 1.5)
        model.bn.bias.normal_(0, 0.3)
    params = model.to_params(
        labels=list(LABELS), groups=list(GROUPS),
        thresholds=np.linspace(0.2, 0.8, len(LABELS)).astype(np.float32),
        trained_labels=np.arange(len(LABELS)) % 5 != 0,
        body_model="fake", neck_feat="after", hparams={"hidden": hidden}, train_log=[{"epoch": 0}],
    )  # fmt: skip
    return model, params


def test_numpy_predict_matches_torch_eval():
    model, params = _trained_like_head()
    x = np.random.default_rng(1).normal(size=(40, 24)).astype(np.float32)
    with torch.no_grad():
        ref = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    np.testing.assert_allclose(hd.predict(params, x), ref, atol=1e-5)
    xh = x.astype(np.float16)  # stored embeddings are fp16
    with torch.no_grad():
        ref16 = torch.sigmoid(model(torch.from_numpy(xh.astype(np.float32)))).numpy()
    np.testing.assert_allclose(hd.predict(params, xh), ref16, atol=1e-5)


def test_head_npz_round_trip(tmp_path):
    _, params = _trained_like_head()
    path = hd.save_head(tmp_path / "h", params)
    assert path == tmp_path / "h" / "head.npz"
    back = hd.load_head(tmp_path / "h")  # folder or file
    assert hd.load_head(path).labels == back.labels
    for k in ("W1", "b1", "bn_mean", "bn_var", "bn_weight", "bn_bias", "W2", "b2", "thresholds"):
        np.testing.assert_array_equal(getattr(back, k), getattr(params, k))
    assert back.labels == list(LABELS) and back.groups == list(GROUPS)
    assert back.trained_labels.tolist() == params.trained_labels.tolist()
    assert back.body_model == "fake" and back.neck_feat == "after"
    assert back.hparams == {"hidden": 16} and back.train_log == [{"epoch": 0}]
    assert back.in_dim == 24
    x = np.random.default_rng(2).normal(size=(5, 24)).astype(np.float32)
    np.testing.assert_array_equal(hd.predict(back, x), hd.predict(params, x))
