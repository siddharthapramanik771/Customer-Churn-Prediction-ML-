from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainingSettings:
    """Settings that control reproducibility, validation, and search behavior."""

    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 3
    scoring: str = "roc_auc"
    grid_search_jobs: int = 1
    grid_search_verbose: int = 1
    param_grid: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "model__n_estimators": [100, 200],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.01, 0.05],
        }
    )
