"""Optuna hyperparameter search of the attribute head (``reid attributes hpo``).

The embeddings are frozen, so one trial is a ~20 s training run of the MLP head on stored
vectors. The data is loaded once per process and reused by every trial. Trials are ranked by the
validation score of :mod:`reid.attributes.train` (mean over datasets of the validation mA without
the ``view_*`` labels); the test split is not read here. TPE proposes parameters (it models good
and bad regions of the space from earlier trials), a median pruner stops trials whose validation
curve is below the median of earlier trials at the same epoch.

Several processes (one per GPU) share one study through a Optuna journal file, so ``--trials``
and ``--timeout-min`` are per process. Optuna is an optional dependency (``reid[train]``) and is
imported inside the functions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from reid.attributes.train import TrainConfig, TrainData, load_train_data, train_head

log = logging.getLogger(__name__)
JOURNAL = "optuna_journal.log"
STUDY = "attr_hpo"
EPOCHS, PATIENCE = 40, 8
MAX_FAILED = 5  # stop a worker when this many trials failed and none completed
_DEFAULT = TrainConfig()
#: The step-1 configuration, enqueued as trial 0 so the baseline uses the same objective.
DEFAULT_PARAMS: dict[str, Any] = {
    k: getattr(_DEFAULT, k)
    for k in ("hidden", "dropout", "lr", "wd", "batch", "peta_weight")
    + ("weighting", "use_flips", "train_view")
}


def require_optuna():  # noqa: ANN201
    """Import optuna or raise an error that says how to install it."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is not installed: run `pip install 'reid[train]'` (or `pip install optuna`)"
        ) from exc
    return optuna


def make_storage(out: str | Path):  # noqa: ANN201
    """Journal-file storage ``<out>/optuna_journal.log`` (safe for several local processes)."""
    optuna = require_optuna()
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    journal = str(path / JOURNAL)
    try:
        from optuna.storages.journal import JournalFileBackend

        backend = JournalFileBackend(journal)
    except ImportError:  # Optuna < 4: the older name
        backend = optuna.storages.JournalFileStorage(journal)
    return optuna.storages.JournalStorage(backend)


def suggest(trial) -> dict[str, Any]:  # noqa: ANN001
    """Search space of the head (epochs 40 and patience 8 are fixed)."""
    return {
        "hidden": trial.suggest_categorical("hidden", [256, 512, 1024, 2048]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.7),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-1, log=True),
        "batch": trial.suggest_categorical("batch", [256, 512, 1024]),
        "peta_weight": trial.suggest_float("peta_weight", 0.5, 8.0, log=True),
        "weighting": trial.suggest_categorical("weighting", ["ratio", "none"]),
        "use_flips": trial.suggest_categorical("use_flips", [True, False]),
        "train_view": trial.suggest_categorical("train_view", [True, False]),
    }


def config_from_params(params: dict[str, Any], seed: int = 0) -> TrainConfig:
    """:class:`TrainConfig` of a trial's parameters (HPO epochs / patience, given seed)."""
    return replace(TrainConfig(), **params, epochs=EPOCHS, patience=PATIENCE, seed=seed)


def short_params(params: dict[str, Any]) -> str:
    """Compact one-line form of a trial's parameters for the ``TRIAL |`` log line."""
    out = []
    for k, v in params.items():
        if isinstance(v, bool):
            v = int(v)
        out.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
    return " ".join(out)


def _trial_line(study, trial) -> str:  # noqa: ANN001
    optuna = require_optuna()
    done = [t.value for t in study.get_trials(deepcopy=False) if t.value is not None]
    best = max(done) if done else float("nan")
    value = "-" if trial.value is None else f"{trial.value:.4f}"
    secs = trial.user_attrs.get("wall_s", (trial.duration.total_seconds() if trial.duration else 0))
    state = trial.state.name if isinstance(trial.state, optuna.trial.TrialState) else trial.state
    return (
        f"TRIAL | #{trial.number} {state} value {value} best {best:.4f} "
        f"epoch {trial.user_attrs.get('best_epoch', '-')} {secs:.0f}s "
        f"| {short_params(trial.params)}"
    )


def make_objective(data: TrainData):  # noqa: ANN201
    """Objective of one trial: train on the shared ``data``, return the best validation score."""
    optuna = require_optuna()

    def objective(trial) -> float:  # noqa: ANN001
        import torch

        params = suggest(trial)
        t0 = time.perf_counter()

        def on_epoch(epoch: int, score: float) -> None:
            if score != score:  # NaN: nothing to report
                return
            trial.report(score, epoch)
            if trial.should_prune():
                trial.set_user_attr("pruned_epoch", epoch)
                trial.set_user_attr("wall_s", time.perf_counter() - t0)
                raise optuna.TrialPruned()

        try:
            head = train_head(data, config_from_params(params), on_epoch=on_epoch)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        hp = head.hparams
        trial.set_user_attr("best_epoch", hp["best_epoch"])
        trial.set_user_attr("val_ma_by_dataset", hp["best_val_ma_by_dataset"])
        trial.set_user_attr("epochs_run", len(head.train_log))
        trial.set_user_attr("wall_s", time.perf_counter() - t0)
        return float(hp["best_val_ma"])

    return objective


def create_study(out: str | Path, name: str = STUDY, seed: int = 0, worker: int = 0):  # noqa: ANN201
    """Create (or load) the shared study: TPE seeded per worker, median pruner, maximize."""
    optuna = require_optuna()
    return optuna.create_study(
        study_name=name,
        storage=make_storage(out),
        sampler=optuna.samplers.TPESampler(seed=seed + worker),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=3),
        direction="maximize",
        load_if_exists=True,
    )


def run_hpo(
    emb_dir: str | Path,
    out: str | Path,
    *,
    name: str = STUDY,
    trials: int = 100,
    timeout_min: float | None = None,
    seed: int = 0,
    worker: int = 0,
    data: TrainData | None = None,
):  # noqa: ANN201
    """Run ``trials`` trials in this process (at most ``timeout_min`` minutes); returns the study.

    Worker 0 enqueues the step-1 default configuration as the first trial when the study is
    still empty. ``data`` (default: loaded from ``emb_dir``) is shared by all trials.
    """
    optuna = require_optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # our own TRIAL | lines instead
    study = create_study(out, name, seed, worker)
    if worker == 0 and not study.get_trials(deepcopy=False):
        study.enqueue_trial(DEFAULT_PARAMS)
        log.info("enqueued the default configuration as the first trial")
    data = load_train_data(emb_dir) if data is None else data
    objective = make_objective(data)

    def report(study, trial) -> None:  # noqa: ANN001
        log.info(_trial_line(study, trial))
        states = [t.state for t in study.get_trials(deepcopy=False)]
        if states.count(optuna.trial.TrialState.FAIL) >= MAX_FAILED and not any(
            s == optuna.trial.TrialState.COMPLETE for s in states
        ):
            log.error("%d failed trials and none completed: stopping this worker", MAX_FAILED)
            study.stop()

    train_log = logging.getLogger("reid.attributes.train")
    old_level = train_log.level
    train_log.setLevel(logging.WARNING)  # one line per epoch x trial would drown the TRIAL lines
    try:
        study.optimize(
            objective, n_trials=trials, timeout=None if not timeout_min else timeout_min * 60,
            callbacks=[report], catch=(Exception,),
        )  # fmt: skip
    finally:
        train_log.setLevel(old_level)
    return study
