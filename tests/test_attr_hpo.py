"""Optuna search and the final comparison on tiny synthetic embeddings (CPU, journal storage)."""

import pytest

torch = pytest.importorskip("torch")
optuna = pytest.importorskip("optuna")

from reid.attributes import hpo  # noqa: E402
from reid.attributes.hpo_report import run_hpo_final  # noqa: E402
from reid.cli import main  # noqa: E402
from tests.test_attr_train import make_all  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_fast(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hpo, "EPOCHS", 5)
    monkeypatch.setattr(hpo, "PATIENCE", 3)


@pytest.fixture(scope="module")
def emb(tmp_path_factory):
    path = tmp_path_factory.mktemp("emb")
    make_all(path)
    return path


def test_search_space_and_default_params():
    assert hpo.DEFAULT_PARAMS == {
        "hidden": 512, "dropout": 0.3, "lr": 1e-3, "wd": 1e-4, "batch": 512,
        "peta_weight": 1.0, "weighting": "ratio", "use_flips": True, "train_view": True,
    }  # fmt: skip
    study = optuna.create_study()
    trial = study.ask()
    params = hpo.suggest(trial)
    assert set(params) == set(hpo.DEFAULT_PARAMS)
    cfg = hpo.config_from_params(params)
    assert (cfg.epochs, cfg.patience) == (hpo.EPOCHS, hpo.PATIENCE)


def test_hpo_enqueues_default_and_study_continues(emb, tmp_path, caplog):
    out = tmp_path / "hpo"
    with caplog.at_level("INFO", logger="reid.attributes.hpo"):
        study = hpo.run_hpo(emb, out, trials=4, seed=0)
    assert len(study.trials) == 4
    first = study.trials[0]
    assert first.params == hpo.DEFAULT_PARAMS and first.state.name == "COMPLETE"
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    assert done and all(0.5 < t.value <= 1.0 for t in done)
    for t in done:
        assert {"best_epoch", "val_ma_by_dataset", "wall_s"} <= set(t.user_attrs)
        assert set(t.user_attrs["val_ma_by_dataset"]) == {"pa100k", "peta"}
    assert sum("TRIAL |" in r.message for r in caplog.records) == 4
    assert (out / hpo.JOURNAL).is_file()
    again = hpo.run_hpo(emb, out, trials=2, seed=0, worker=1)  # same journal: continues
    assert len(again.trials) == 6
    assert (
        len([t for t in again.trials if t.params == hpo.DEFAULT_PARAMS]) == 1
    )  # not enqueued twice


def test_hpo_final_writes_report_files(emb, tmp_path):
    out = tmp_path / "hpo"
    hpo.run_hpo(emb, out, trials=3)
    path = run_hpo_final(emb, out, seeds=2)
    assert path.name == "RESULTS_hpo.md"
    text = path.read_text()
    for part in ("Default vs tuned", "Viewpoint ablation", "Parameter importances", "+-"):
        assert part in text
    assert "3 trials" in text and "seeds 0..1" in text
    assert (out / "head_tuned" / "head.npz").is_file()
    assert (out / "trials.csv").is_file()
    assert (out / "report_tuned" / "RESULTS_attributes_tuned.md").is_file()


def test_cli_hpo_and_hpo_final(emb, tmp_path, capsys):
    out = tmp_path / "hpo"
    base = ["attributes", "hpo", "--emb", str(emb), "--out", str(out)]
    assert main([*base, "--trials", "2", "--timeout-min", "5", "--worker", "0"]) == 0
    assert "2 trials" in capsys.readouterr().out
    assert main(["attributes", "hpo-final", "--emb", str(emb), "--out", str(out)]) == 0
    assert (out / "RESULTS_hpo.md").is_file()
    assert main(["attributes", "hpo-final", "--emb", str(emb), "--out", str(tmp_path / "x")]) == 1
