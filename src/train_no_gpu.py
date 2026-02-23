import os
import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_no_gpu(
    data_path: Optional[str] = None,
    target_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    n_estimators: Optional[int] = None,
    mlflow_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Train a CPU-only sklearn model and track with MLflow.

    Configuration is read from environment variables when available:
    - MLFLOW_TRACKING_URI: URL or path for mlflow tracking server
    - MLFLOW_EXPERIMENT_NAME: experiment name to use
    - MLFLOW_RUN_NAME: optional run name
    - RANDOM_SEED: integer seed
    - N_ESTIMATORS: number of trees for RandomForest

    Returns a dict with the trained model, metrics, params and mlflow run id.
    """

    # Merge mlflow_env with os.environ (mlflow_env overrides os.environ if provided)
    env = dict(os.environ)
    if mlflow_env:
        env.update(mlflow_env)

    # Configure mlflow if environment variables are present
    tracking_uri = env.get("MLFLOW_TRACKING_URI")
    experiment_name = env.get("MLFLOW_EXPERIMENT_NAME", "default")
    run_name = env.get("MLFLOW_RUN_NAME")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to {tracking_uri}")

    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to {experiment_name}")

    # Parameters with precedence: function args -> env vars -> defaults
    seed = (
        int(random_state)
        if random_state is not None
        else int(env.get("RANDOM_SEED", "42"))
    )
    n_estimators_final = (
        int(n_estimators)
        if n_estimators is not None
        else int(env.get("N_ESTIMATORS", "100"))
    )

    # Load data: CSV if provided, otherwise iris dataset
    if data_path:
        df = pd.read_csv(data_path)
        if target_col is None:
            raise ValueError("When providing data_path you must also provide target_col")
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
    else:
        X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(n_estimators=n_estimators_final, random_state=seed, n_jobs=1),
            ),
        ]
    )

    with mlflow.start_run(run_name=run_name) as run:
        # Log params
        mlflow.log_param("n_estimators", n_estimators_final)
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("test_size", test_size)

        # Fit
        pipeline.fit(X_train, y_train)

        # Predict and evaluate
        preds = pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="macro"))

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        # Log the model
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        result = {
            "model": pipeline,
            "metrics": {"accuracy": acc, "f1_macro": f1},
            "params": {"n_estimators": n_estimators_final, "random_seed": seed, "test_size": test_size},
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
        }

    return result


if __name__ == "__main__":
    # Example run when executed as a script.
    # Configure MLflow via environment variables if desired.
    res = train_no_gpu()
    print("Training finished. Summary:")
    print(f"run_id: {res['run_id']}")
    print(f"metrics: {res['metrics']}")
