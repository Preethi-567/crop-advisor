"""
main.py
-------
Entry point for the Crop Recommendation System.
Run this file to train, evaluate, and save the model.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import CropDataPreprocessor
from model_training      import ModelTrainer
from predict             import CropPredictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
DATA_PATH       = "Crop_recommendation.csv"
MODEL_PATH      = "models/crop_model.pkl"
ENCODER_PATH    = "models/label_encoder.pkl"
REPORTS_DIR     = "reports"

os.makedirs("models",  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def main():
    # ── 1. Load & Validate Data ────────────────────────────────────────────────
    preprocessor = CropDataPreprocessor()
    df = preprocessor.load_and_validate(DATA_PATH)

    X_train, X_test, y_train, y_test = preprocessor.prepare_features(df)

    # ── 2. Train & Compare All Models ─────────────────────────────────────────
    trainer = ModelTrainer(cv_folds=5)
    results_df = trainer.compare_models(X_train, y_train, X_test, y_test)

    print("\n" + "="*65)
    print("          MODEL COMPARISON RESULTS")
    print("="*65)
    print(results_df.to_string(index=False))
    print("="*65)

    # ── 3. Hyperparameter Tuning on Random Forest ──────────────────────────────
    logger.info("Tuning Random Forest…")
    tuned_rf = trainer.tune_random_forest(X_train, y_train)

    # ── 4. Full Evaluation of Best Model ──────────────────────────────────────
    class_names = list(preprocessor.label_encoder.classes_)
    eval_results = trainer.full_evaluation(
        trainer.best_model, X_test, y_test, class_names
    )

    print(f"\n✅  Best Model: {trainer.best_model_name}")
    print(f"    Test Accuracy: {eval_results['accuracy']:.4f}")

    # ── 5. Save Model + Encoder ────────────────────────────────────────────────
    trainer.save_model(trainer.best_model, MODEL_PATH)
    joblib.dump(preprocessor.label_encoder, ENCODER_PATH)
    logger.info(f"Label encoder saved → {ENCODER_PATH}")

    # ── 6. Quick Demo Prediction ───────────────────────────────────────────────
    predictor = CropPredictor(trainer.best_model, preprocessor.label_encoder)

    sample = {"N": 90, "P": 42, "K": 43, "temperature": 20.9,
              "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}

    print("\n" + "─"*55)
    print("  DEMO PREDICTION")
    print("─"*55)
    print(f"  Input: {sample}")
    recommendations = predictor.top_n_recommendations(sample, n=3)
    for rec in recommendations:
        print(f"\n  #{rec['rank']} → {rec['crop']}  ({rec['confidence']})")
        print(f"       Why: {rec['reason']}")
    print("─"*55)

    # ── 7. Save Comparison Plot ────────────────────────────────────────────────
    _plot_model_comparison(results_df)

    logger.info("Pipeline complete. All artifacts saved.")


def _plot_model_comparison(results_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Crop Recommendation — Model Comparison", fontsize=14, fontweight="bold")

    # Accuracy bars
    ax = axes[0]
    colors = ["#2ecc71" if v == results_df["Test Acc"].max() else "#3498db"
              for v in results_df["Test Acc"]]
    ax.barh(results_df["Model"], results_df["Test Acc"], color=colors)
    ax.set_xlabel("Test Accuracy")
    ax.set_xlim(0.85, 1.01)
    ax.set_title("Test Accuracy by Model")
    for i, v in enumerate(results_df["Test Acc"]):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

    # CV Mean ± Std
    ax2 = axes[1]
    ax2.barh(results_df["Model"], results_df["CV Mean"],
             xerr=results_df["CV Std"], color="#9b59b6", alpha=0.8,
             error_kw={"ecolor": "black", "capsize": 4})
    ax2.set_xlabel("CV Accuracy (mean ± std)")
    ax2.set_xlim(0.85, 1.05)
    ax2.set_title("5-Fold Cross-Validation Accuracy")

    plt.tight_layout()
    path = f"{REPORTS_DIR}/model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Plot saved → {path}")


if __name__ == "__main__":
    main()
