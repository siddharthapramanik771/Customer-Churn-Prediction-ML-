import joblib
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_PATH,
    TARGET_COLUMN,
    ensure_runtime_dirs,
    get_mlflow_tracking_uri,
    load_dataset,
)
from preprocessing import build_preprocessor, clean, encode_target

def main():
    ensure_runtime_dirs()
    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    df = load_dataset(DATA_PATH)
    df = clean(df)

    X = df.drop(TARGET_COLUMN, axis=1)
    y = encode_target(df[TARGET_COLUMN])

    pre, _, _ = build_preprocessor(df)

    smote = SMOTE()
    # Fit preprocessor first for SMOTE to work on numeric space
    X_enc = pre.fit_transform(X)
    X_res, y_res = smote.fit_resample(X_enc, y)

    pipe = Pipeline([
        ('model', XGBClassifier(tree_method="hist", eval_metric="logloss"))
    ])

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [4, 6],
        'model__learning_rate': [0.01, 0.05]
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        grid = GridSearchCV(
            pipe,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            error_score='raise',
        )
        grid.fit(X_res, y_res)

        best_model = grid.best_estimator_
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_score", grid.best_score_)

        # save preprocessor + model together
        bundle = {"preprocessor": pre, "model": best_model}
        joblib.dump(bundle, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)

        print("Saved model to", MODEL_PATH)

if __name__ == "__main__":
    main()
