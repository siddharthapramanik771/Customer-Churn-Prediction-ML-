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
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [2, 3, 4, 5],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__min_child_weight": [1, 3, 5],
        }
    )
