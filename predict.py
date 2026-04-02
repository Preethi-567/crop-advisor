"""
predict.py
----------
Inference engine for the Crop Recommendation System.
Handles single predictions, confidence scores, top-N recommendations,
and SHAP-based explainability.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# Agronomic reasoning templates
# WHY add reasoning? 
# Pure predictions are black boxes. Explaining WHY a crop is recommended
# builds farmer trust and makes your project 10x more impressive in demos.

CROP_REASONING = {
    "rice":        "Thrives in high humidity (>80%) and substantial rainfall (>200mm) with moderate nitrogen.",
    "maize":       "Prefers warm temperatures (20–30°C) and moderate potassium levels.",
    "jute":        "Requires very high humidity and rainfall; grows well in loamy soil.",
    "cotton":      "Grows in warm, dry conditions with high potassium and moderate phosphorus.",
    "coconut":     "Needs tropical humidity and temperatures above 27°C.",
    "papaya":      "Favours warm temperatures, moderate rainfall, and slightly acidic soil.",
    "orange":      "Prefers cool winters and warm summers; optimal pH 5.5–6.5.",
    "apple":       "Needs cool temperatures (<25°C) and low humidity; high rainfall tolerant.",
    "muskmelon":   "Grows best in hot, dry climates with sandy soil and moderate K.",
    "watermelon":  "Warm-season crop needing high temperatures and low humidity.",
    "grapes":      "Requires warm, dry summers; sensitive to high rainfall during ripening.",
    "mango":       "Thrives in tropical climates with dry cool winters and hot summers.",
    "banana":      "High humidity, warm temperatures, and fertile soils with high N and K.",
    "pomegranate": "Drought-tolerant; prefers semi-arid conditions and neutral pH.",
    "lentil":      "Cool-season crop needing low humidity and moderate N-fixing soil.",
    "blackgram":   "Warm, moist conditions; effective nitrogen fixer reducing N requirements.",
    "mungbean":    "Short-duration crop; prefers warm temperatures and well-drained soils.",
    "mothbeans":   "Extremely drought-tolerant; thrives in arid/semi-arid conditions.",
    "pigeonpeas":  "Nitrogen-fixing legume; grows in tropical conditions with variable rainfall.",
    "kidneybeans": "Cool climate crop; needs well-drained soil and moderate rainfall.",
    "chickpea":    "Drought-tolerant legume; prefers cool, dry growing season.",
    "coffee":      "Grows in tropical highlands with stable temperatures and high humidity.",
}


class CropPredictor:
    """
    Inference wrapper for the trained crop recommendation pipeline.
    Provides predictions, confidence scores, top-N crops, and explanations.
    """

    def __init__(self, model: Pipeline, label_encoder: LabelEncoder):
        self.model         = model
        self.label_encoder = label_encoder

    # ── Core Prediction ────────────────────────────────────────────────────────
    def predict(self, features: Dict[str, float]) -> str:
        """
        Predict the single best crop for the given soil/climate features.

        Parameters
        ----------
        features : dict with keys N, P, K, temperature, humidity, ph, rainfall

        Returns
        -------
        Crop name as string.
        """
        X = self._dict_to_frame(features)
        pred_enc = self.model.predict(X)[0]
        return self.label_encoder.inverse_transform([pred_enc])[0]

    def predict_with_confidence(
        self, features: Dict[str, float]
    ) -> Tuple[str, float]:
        """
        Returns (best_crop, confidence_percentage).
        Only works if the underlying model supports predict_proba.
        """
        X = self._dict_to_frame(features)
        proba = self.model.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        best_crop = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = round(proba[pred_idx] * 100, 2)
        return best_crop, confidence

    def top_n_recommendations(
        self, features: Dict[str, float], n: int = 3
    ) -> List[Dict]:
        """
        Return top-N crop recommendations with probabilities and reasoning.

        WHY top-N?
        Farmers may have constraints (market availability, seeds, equipment)
        that make the single top crop unsuitable. Offering alternatives
        is a practical, production-grade feature.
        """
        X = self._dict_to_frame(features)
        proba = self.model.predict_proba(X)[0]
        top_indices = np.argsort(proba)[::-1][:n]

        results = []
        for idx in top_indices:
            crop = self.label_encoder.inverse_transform([idx])[0]
            results.append({
                "rank":       len(results) + 1,
                "crop":       crop.capitalize(),
                "confidence": f"{proba[idx]*100:.1f}%",
                "reason":     CROP_REASONING.get(crop, "Suitable based on soil and climate data."),
            })
        return results

    # ── SHAP Explainability ────────────────────────────────────────────────────
    def explain_with_shap(
        self,
        X_train: pd.DataFrame,
        X_explain: pd.DataFrame,
        model_step_name: str = "clf"
    ):
        """
        Use SHAP TreeExplainer to show which features drove the prediction.

        WHY SHAP?
        SHAP (SHapley Additive exPlanations) gives each feature a fair
        credit for how much it pushed the prediction toward or away from
        each class. Unlike feature importance, SHAP values are directional
        and additive — i.e. you can say "high Rainfall pushed toward Rice
        by +0.34 probability units."

        Only works with tree-based models (RF, GBM). For GNB/LR, use
        KernelExplainer (slower but model-agnostic).
        """
        try:
            import shap

            clf = self.model.named_steps[model_step_name]

            # Transform through any preprocessing steps before the clf
            pipe_steps = list(self.model.named_steps.keys())
            pre_steps  = pipe_steps[: pipe_steps.index(model_step_name)]
            X_tr_t = X_train.copy()
            X_ex_t = X_explain.copy()
            for step in pre_steps:
                transformer = self.model.named_steps[step]
                X_tr_t = transformer.transform(X_tr_t)
                X_ex_t = transformer.transform(X_ex_t)

            explainer  = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_ex_t)
            return explainer, shap_values, X_ex_t

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return None, None, None

    # ── Helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _dict_to_frame(features: Dict[str, float]) -> pd.DataFrame:
        """Convert an input dict to a single-row DataFrame for sklearn."""
        cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        return pd.DataFrame([[features[c] for c in cols]], columns=cols)

    @staticmethod
    def load(model_path: str, encoder_path: str) -> "CropPredictor":
        """Load a persisted predictor from disk."""
        model   = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        return CropPredictor(model, encoder)