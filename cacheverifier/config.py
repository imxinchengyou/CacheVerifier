from dataclasses import dataclass, field
from pathlib import Path

import yaml

# vCache's own benchmark.py STATIC_THRESHOLDS / DELTAS grids, so Group A/B
# sweeps line up with the exact points used to produce the paper's figures.
DEFAULT_THRESHOLD_GRID = [0.80, 0.83, 0.86, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99]
DEFAULT_TARGET_ERROR_RATES = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07]


@dataclass
class DatasetConfig:
    """Declarative experiment settings for one benchmark trace, so the exact
    knobs used to produce a set of numbers are checked into `configs/`
    rather than only living in someone's shell history.

    No `seed` / `history_fraction`: matching vCache's own evaluation
    protocol, there is no offline history split — `max_samples` records are
    streamed once, in the dataset's original order, through an initially
    empty cache (see `cacheverifier.experiments.runner.ExperimentRunner`).
    """

    name: str
    processed_path: str
    max_samples: int | None = None
    embedder: str = "hash"
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    threshold_grid: list[float] = field(default_factory=lambda: list(DEFAULT_THRESHOLD_GRID))
    target_error_rates: list[float] = field(default_factory=lambda: list(DEFAULT_TARGET_ERROR_RATES))


def load_dataset_config(path: str | Path) -> DatasetConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DatasetConfig(**raw)
