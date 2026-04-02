"""
model_training.py
-----------------
All model definitions, training, cross-validation,
hyperparameter tuning, and evaluation in one clean module.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import joblib
import logging
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)


# ─── Model Registry ──────────────────────────────────────────────────────────
# WHY Pipelines?
# A Pipeline bundles preprocessing + model into ONE object.
# This prevents data leakage: the scaler is fit ONLY on training data,
# even inside cross-validation folds. Without a pipeline, if you scale
# all data first and then CV, test-fold information leaks into the scaler.

def build_model_registry() -> Dict[str, Pipeline]:
    """Return a dictionary of all candidate models wrapped in sklearn Pipelines."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GaussianNB())
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(probability=True, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("clf",    DecisionTreeClassifier(random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("clf",    RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("clf",    GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]),
    }


class ModelTrainer:
    """
    Trains multiple models, cross-validates them, and selects the best one.
    Supports hyperparameter tuning and model persistence.
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds     = cv_folds
        self.random_state = random_state
        self.cv           = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        self.results_df: pd.DataFrame = pd.DataFrame()
        self.best_model_name: str = ""
        self.best_model: Pipeline = None

    # ── Cross-Validation Benchmark ─────────────────────────────────────────────
    def compare_models(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test:  pd.DataFrame,
        y_test:  np.ndarray
    ) -> pd.DataFrame:
        """
        Run all models through 5-fold stratified CV and evaluate on test set.

        WHY Stratified K-Fold?
        In multi-class problems, plain KFold can produce folds with
        missing classes. StratifiedKFold ensures every fold mirrors the
        original class distribution — essential for reliable accuracy estimates.
        """
        registry = build_model_registry()
        rows = []

        for name, pipeline in registry.items():
            logger.info(f"Training: {name}")

            # Cross-validation on training data
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=self.cv, scoring="accuracy", n_jobs=-1
            )

            # Final fit + test evaluation
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            row = {
                "Model":       name,
                "CV Mean":     round(cv_scores.mean(), 4),
                "CV Std":      round(cv_scores.std(),  4),
                "Test Acc":    round(accuracy_score(y_test, y_pred),                          4),
                "Precision":   round(precision_score(y_test, y_pred, average="weighted"),     4),
                "Recall":      round(recall_score(y_test, y_pred,    average="weighted"),     4),
                "F1-Score":    round(f1_score(y_test, y_pred,        average="weighted"),     4),
            }
            rows.append(row)

        self.results_df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)

        # Store the best pipeline object
        best_row = self.results_df.iloc[0]
        self.best_model_name = best_row["Model"]
        self.best_model      = registry[self.best_model_name]
        self.best_model.fit(X_train, y_train)

        logger.info(f"Best model: {self.best_model_name} (F1={best_row['F1-Score']})")
        return self.results_df

    # ── Hyperparameter Tuning ──────────────────────────────────────────────────
    def tune_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray
    ) -> Pipeline:
        """
        GridSearchCV for Random Forest hyperparameters.

        WHY tune?
        Default hyperparameters are rarely optimal. Tuning n_estimators
        and max_depth controls bias-variance tradeoff, while min_samples_leaf
        prevents overfitting on small clusters of crops.
        """
        logger.info("Running GridSearchCV on Random Forest…")

        param_grid = {
            "clf__n_estimators":    [100, 200],
            "clf__max_depth":       [None, 10, 20],
            "clf__min_samples_leaf":[1, 2, 4],
        }

        rf_pipe = Pipeline([
            ("clf", RandomForestClassifier(random_state=self.random_state))
        ])

        grid = GridSearchCV(
            rf_pipe, param_grid,
            cv=self.cv, scoring="f1_weighted",
            n_jobs=-1, verbose=0
        )
        grid.fit(X_train, y_train)

        logger.info(f"Best RF params: {grid.best_params_}")
        logger.info(f"Best RF CV F1:  {grid.best_score_:.4f}")
        return grid.best_estimator_

    # ── Detailed Evaluation ────────────────────────────────────────────────────
    def full_evaluation(
        self,
        model: Pipeline,
        X_test:  pd.DataFrame,
        y_test:  np.ndarray,
        class_names: list
    ) -> Dict[str, Any]:
        """
        Return accuracy, full classification report, and confusion matrix
        for a fitted model.
        """
        y_pred = model.predict(X_test)

        return {
            "accuracy":             accuracy_score(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, target_names=class_names, output_dict=True
            ),
            "confusion_matrix":     confusion_matrix(y_test, y_pred),
            "y_pred":               y_pred,
        }

    # ── Persistence ────────────────────────────────────────────────────────────
    @staticmethod
    def save_model(pipeline: Pipeline, path: str) -> None:
        """
        Persist the entire sklearn Pipeline (scaler + model) to disk.

        WHY save the whole pipeline?
        Saving just the model weights without the scaler would force you
        to manually scale at inference time — error-prone. One pipeline
        object = one .pkl file = zero mistakes.
        """
        joblib.dump(pipeline, path)
        logger.info(f"Model saved → {path}")

    @staticmethod
    def load_model(path: str) -> Pipeline:
        """Load a persisted pipeline."""
        pipeline = joblib.load(path)
        logger.info(f"Model loaded ← {path}")
        return pipeline