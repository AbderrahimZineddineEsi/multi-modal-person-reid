"""``reid attributes hpo-final``: read the HPO study, retrain default vs best on several seeds.

Run once after the HPO processes finished. This is the only place where the **test** split is
read: the default and the best configuration are trained with seeds ``0..n-1`` and scored on the
test splits (mean +- std over seeds), and the best configuration is also trained with and without
viewpoint labels (viewpoint ablation). All comparisons use the non-``view_*`` labels, the same
label set as the HPO objective, so a head trained without view labels is comparable with one
trained with them. The seed-0 head of the best configuration is saved as ``head_tuned`` and
reported with the step-1 report (``report_tuned``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from reid.attributes import store
from reid.attributes.head import HeadParams, predict, save_head
from reid.attributes.hpo import (
    DEFAULT_PARAMS,
    STUDY,
    config_from_params,
    make_storage,
    require_optuna,
)
from reid.attributes.report import run_report
from reid.attributes.schema import PA100K_COLUMNS
from reid.attributes.train import DATASETS, VIEW_COLS, TrainData, load_train_data, train_head
from reid.eval.par_metrics import instance_metrics, mean_accuracy

log = logging.getLogger(__name__)


def load_tests(emb_dir: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``dataset -> (embeddings, targets, evaluated label columns)`` of the test splits."""
    out = {}
    for name in DATASETS:
        if "test" not in store.available_splits(emb_dir, name):
            continue
        sp = store.load_split(emb_dir, name, "test", include_flips=False)
        cols = (
            np.array(PA100K_COLUMNS)
            if name == "pa100k"
            else np.flatnonzero((sp.targets != -1).any(0))
        )
        out[name] = (sp.x, sp.targets, cols)
    if not out:
        raise FileNotFoundError(f"no test embeddings in {emb_dir}")
    return out


def evaluate_head(head: HeadParams, tests: dict) -> dict[str, dict[str, float]]:
    """Test metrics per dataset over the trained non-view labels, plus mA over all trained."""
    res = {}
    for name, (x, t, cols) in tests.items():
        probs = predict(head, x)
        trained = cols[head.trained_labels[cols]]
        nv = trained[~np.isin(trained, VIEW_COLS)]
        res[name] = {
            "ma": mean_accuracy(probs, t, 0.5, nv),
            "ma_tuned": mean_accuracy(probs, t, head.thresholds, nv),
            "f1": instance_metrics(probs, t, 0.5, nv)["f1"],
            "ma_all": mean_accuracy(probs, t, 0.5, trained),
        }
    return res


def _train_seeds(
    data: TrainData, params: dict[str, Any], seeds: int, tests: dict, label: str
) -> tuple[list[dict], HeadParams]:
    """Train ``seeds`` heads of one configuration; returns their test metrics and seed 0's head."""
    results, first = [], None
    for seed in range(seeds):
        head = train_head(data, config_from_params(params, seed))
        first = first or head
        results.append(evaluate_head(head, tests))
        log.info("%s seed %d: %s", label, seed, results[-1])
    return results, first


def _ms(runs: list[dict], ds: str, key: str) -> str:
    v = 100 * np.array([r[ds][key] for r in runs], float)
    return f"{v.mean():.2f} +- {v.std(ddof=1) if len(v) > 1 else 0.0:.2f}"


def _compare_table(rows: dict[str, list[dict]], datasets: list[str]) -> list[str]:
    out = ["| config | dataset | mA (0.5) | F1 (0.5) | mA (tuned thr.) | mA (all trained) |"]
    out.append("|---|---|---|---|---|---|")
    for cfg, runs in rows.items():
        for ds in datasets:
            cells = " | ".join(_ms(runs, ds, k) for k in ("ma", "f1", "ma_tuned", "ma_all"))
            out.append(f"| {cfg} | {ds} | {cells} |")
    return out


def _importances(study) -> dict[str, float]:  # noqa: ANN001
    """Parameter importances (Optuna's default evaluator); {} when it cannot be computed."""
    optuna = require_optuna()
    try:
        imp = dict(optuna.importance.get_param_importances(study))
    except Exception as exc:  # noqa: BLE001 - too few trials, constant objective, ...
        log.warning("parameter importances skipped: %s", exc)
        return {}
    return imp


def _plots(study, out: Path) -> list[str]:  # noqa: ANN001
    """Optimisation history and importance PNGs (optuna.visualization.matplotlib, Agg)."""
    optuna = require_optuna()
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    import warnings

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    try:
        from optuna.visualization import matplotlib as vis
    except Exception as exc:  # noqa: BLE001
        log.warning("optuna plots skipped: %s", exc)
        return []
    made = []
    for fn, png in (
        (vis.plot_optimization_history, "optimization_history.png"),
        (vis.plot_param_importances, "param_importances.png"),
    ):
        try:
            ax = fn(study)
            fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
            fig.tight_layout()
            fig.savefig(out / png, dpi=110)
            plt.close(fig)
            made.append(png)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s skipped: %s", png, exc)
    return made


def _study_summary(study) -> tuple[dict[str, int], float, float]:  # noqa: ANN001
    """State counts, wall time (first start to last end) and the sum of trial durations."""
    trials = study.get_trials(deepcopy=False)
    counts = {}
    for t in trials:
        counts[t.state.name] = counts.get(t.state.name, 0) + 1
    starts = [t.datetime_start for t in trials if t.datetime_start]
    ends = [t.datetime_complete for t in trials if t.datetime_complete]
    wall = (max(ends) - min(starts)).total_seconds() if starts and ends else 0.0
    busy = sum(t.duration.total_seconds() for t in trials if t.duration)
    return counts, wall, busy


def run_hpo_final(emb_dir: str | Path, out: str | Path, name: str = STUDY, seeds: int = 3) -> Path:
    """Write ``RESULTS_hpo.md``, ``trials.csv``, PNGs, ``head_tuned/`` and ``report_tuned/``."""
    optuna = require_optuna()
    out = Path(out)
    study = optuna.load_study(study_name=name, storage=make_storage(out))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise RuntimeError(f"study {name!r} in {out} has no completed trial")
    best = study.best_trial
    default_trial = next((t for t in done if t.params == DEFAULT_PARAMS), None)
    counts, wall, busy = _study_summary(study)
    data, tests = load_train_data(emb_dir), load_tests(emb_dir)
    datasets = list(tests)
    runs = {}
    runs["default"], _ = _train_seeds(data, DEFAULT_PARAMS, seeds, tests, "default")
    runs["tuned"], tuned_head = _train_seeds(data, best.params, seeds, tests, "tuned")
    abl = {}
    for view in (True, False):
        if best.params["train_view"] == view:
            abl[view] = runs["tuned"]
        else:
            abl[view], _ = _train_seeds(
                data, {**best.params, "train_view": view}, seeds, tests, f"view={view}"
            )
    path = save_head(out / "head_tuned", tuned_head)
    run_report(emb_dir, path, out / "report_tuned", "tuned")
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    imp = _importances(study)
    pngs = _plots(study, out)

    imp_md = ["Not computed (too few trials or a constant objective)."]
    if imp:
        imp_md = ["| parameter | importance |", "|---|---|"]
        imp_md += [f"| {k} | {v:.3f} |" for k, v in imp.items()]
    md = [
        "# Attribute head: hyperparameter search (Optuna)", "",
        f"Study `{name}`: {len(study.trials)} trials "
        f"({counts.get('COMPLETE', 0)} complete, {counts.get('PRUNED', 0)} pruned, "
        f"{counts.get('FAIL', 0)} failed). Wall time {wall / 60:.1f} min "
        f"(sum of trial times {busy / 60:.1f} min, more when several GPUs ran at once).", "",
        f"Best trial #{best.number}: validation score **{100 * best.value:.2f}** "
        f"(mean over datasets of val mA without view labels)"
        + (
            f"; the default configuration scored {100 * default_trial.value:.2f} "
            f"(trial #{default_trial.number})."
            if default_trial
            else "."
        ),
        "", "Best parameters:", "",
        "| parameter | best | default |", "|---|---|---|",
        *(f"| {k} | {v} | {DEFAULT_PARAMS[k]} |" for k, v in best.params.items()),
        "", f"## Default vs tuned on test (mean +- std over seeds 0..{seeds - 1})", "",
        *_compare_table(runs, datasets), "",
        "mA, F1 and tuned-threshold mA are over the trained labels **without** `view_*` "
        "(PA-100K: 23 of its 26 attributes); `mA (all trained)` also counts the view labels "
        "when the head was trained on them. The per-label table of the tuned head is in "
        "`report_tuned/`.", "",
        f"## Viewpoint ablation (best config, seeds 0..{seeds - 1}, test mA without view labels)",
        "", "| training | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets),
        *(
            f"| {'with' if v else 'without'} view labels | "
            + " | ".join(_ms(abl[v], ds, "ma") for ds in datasets)
            + " |"
            for v in (True, False)
        ),
        "", "## Parameter importances", "",
        *imp_md,
        "", *(f"![{p}]({p})" for p in pngs), "",
        "## How to read this", "",
        "Trials were ranked on **validation** only; the test numbers above were read once, after "
        "the search, so they are not optimistic from selection. A difference between default "
        "and tuned that is smaller than the seed-to-seed std is noise. The importances say which "
        "parameters moved the validation score most across trials (an estimate that is rough "
        "with few trials). In the ablation, a gap between the rows is the effect of also "
        "learning front / side / back on the other attributes.",
    ]  # fmt: skip
    res = out / "RESULTS_hpo.md"
    res.write_text("\n".join(md) + "\n")
    return res
