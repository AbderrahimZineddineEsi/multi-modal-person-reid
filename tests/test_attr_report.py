"""``reid attributes train`` / ``report`` end to end on synthetic embeddings (CPU)."""

import json

import numpy as np
import pytest

from reid.cli import main

torch = pytest.importorskip("torch")

from tests.test_attr_train import make_all, make_split  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _peta_with_persons(emb):
    """Replace the PETA test split by one with 20 persons of 6 images and noisy-ish inputs."""
    x, t = make_split(emb, "peta", "test", 120, 77, flips=False)
    persons = np.repeat([f"s/{i}" for i in range(20)], 6)
    np.save(emb / "peta/test/persons.npy", persons)
    return x, t


def test_cli_train_then_report(tmp_path, capsys):
    emb = tmp_path / "emb"
    make_all(emb)
    _peta_with_persons(emb)
    args = ["--hidden", "16", "--epochs", "6", "--patience", "3", "--lr", "5e-3", "--batch", "128"]
    assert (
        main(["attributes", "train", "--emb", str(emb), "--out", str(tmp_path / "head"), *args])
        == 0
    )
    assert (tmp_path / "head" / "head.npz").is_file()
    code = main(["attributes", "report", "--emb", str(emb), "--head", str(tmp_path / "head"),
                 "--out", str(tmp_path / "rep"), "--tag", "t"])  # fmt: skip
    assert code == 0
    rep = tmp_path / "rep"
    for name in ("RESULTS_attributes_t.md", "metrics_t.json", "voting_t.png", "training_t.png"):
        assert (rep / name).is_file(), name
    m = json.loads((rep / "metrics_t.json").read_text())
    assert set(m["datasets"]) == {"pa100k", "peta"}
    peta = m["datasets"]["peta"]
    assert peta["ma"] > 0.6 and peta["labels"]["hat"]["n_pos"] > 0
    assert "stroller" in peta["labels_untrained"]
    assert [r["k"] for r in peta["voting"]][:1] == [1] and peta["voting"][-1]["k"] == 4
    assert "hat" in m["ui_candidates"]
    md = (rep / "RESULTS_attributes_t.md").read_text()
    assert "Per label" in md and "Voting" in md and "80.21" in md


def test_report_without_test_embeddings_fails_cleanly(tmp_path, capsys):
    emb = tmp_path / "emb"
    make_all(emb, with_test=False)
    assert main(["attributes", "train", "--emb", str(emb), "--out", str(tmp_path / "h"),
                 "--epochs", "2", "--hidden", "8"]) == 0  # fmt: skip
    code = main(["attributes", "report", "--emb", str(emb), "--head", str(tmp_path / "h"),
                 "--out", str(tmp_path / "r")])  # fmt: skip
    assert code == 1 and "no test embeddings" in capsys.readouterr().err
