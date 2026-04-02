"""
data_preprocessing.py
---------------------
Handles all data loading, validation, and preprocessing
for the Crop Recommendation System.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN   = "label"

# Realistic agronomic input ranges (based on dataset statistics)
FEATURE_RANGES = {
    "N":           (0,   140),
    "P":           (5,   145),
    "K":           (5,   205),
    "temperature": (8,   44),
    "humidity":    (14,  100),
    "ph":          (3.5, 10.0),
    "rainfall":    (20,  300),
}


class CropDataPreprocessor:
    """
    Loads, validates, and prepares the crop dataset for modelling.

    Attributes
    ----------
    label_encoder : LabelEncoder
        Fitted encoder to convert between crop names and integer labels.
    """

    def __init__(self):
        self.label_encoder = LabelEncoder()

    # ── Public API ─────────────────────────────────────────────────────────────
    def load_and_validate(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV and run sanity checks. Returns a clean DataFrame.

        Why validate?
        Real-world data always has surprises — duplicate rows, missing values,
        wrong dtypes. Catching them early prevents silent model degradation.
        """
        logger.info(f"Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)

        self._check_schema(df)
        self._check_missing(df)
        self._check_duplicates(df)
        self._check_class_balance(df)

        logger.info(f"Dataset validated. Shape: {df.shape}")
        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple:
        """
        Encode labels and split into stratified train/test sets.

        Returns
        -------
        X_train, X_test, y_train, y_test, label_encoder
        """
        X = df[FEATURE_COLUMNS].copy()
        y_raw = df[TARGET_COLUMN]

        # ── LabelEncoder is correct here: it assigns stable integer IDs ──
        # We preserve the mapping via label_encoder.classes_ so predictions
        # can be decoded back to crop names at inference time.
        y = self.label_encoder.fit_transform(y_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y          # ensures every crop appears in both splits
        )

        logger.info(
            f"Split complete → Train: {X_train.shape[0]} | Test: {X_test.shape[0]}"
        )
        return X_train, X_test, y_train, y_test

    def decode_label(self, encoded: int) -> str:
        """Convert integer prediction back to crop name."""
        return self.label_encoder.inverse_transform([encoded])[0]

    # ── Validation helpers ─────────────────────────────────────────────────────
    def _check_schema(self, df: pd.DataFrame) -> None:
        required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

    def _check_missing(self, df: pd.DataFrame) -> None:
        nulls = df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum()
        if nulls.any():
            logger.warning(f"Missing values detected:\n{nulls[nulls > 0]}")
        else:
            logger.info("No missing values found.")

    def _check_duplicates(self, df: pd.DataFrame) -> None:
        n_dups = df.duplicated().sum()
        if n_dups > 0:
            logger.warning(f"{n_dups} duplicate rows found and dropped.")
            df.drop_duplicates(inplace=True)
        else:
            logger.info("No duplicate rows found.")

    def _check_class_balance(self, df: pd.DataFrame) -> None:
        counts = df[TARGET_COLUMN].value_counts()
        logger.info(
            f"Class balance — min: {counts.min()}, max: {counts.max()}, "
            f"classes: {len(counts)}"
        )
        if counts.max() / counts.min() > 3:
            logger.warning(
                "Dataset may be imbalanced. Consider SMOTE or class weights."
            )

    @staticmethod
    def validate_input(features: dict) -> bool:
        """
        Validate a single prediction input against known agronomic ranges.
        Returns True if valid, raises ValueError with a clear message if not.
        """
        for feat, (lo, hi) in FEATURE_RANGES.items():
            val = features.get(feat)
            if val is None:
                raise ValueError(f"Missing feature: {feat}")
            if not (lo <= val <= hi):
                raise ValueError(
                    f"'{feat}' = {val} is outside expected range [{lo}, {hi}]"
                )
        return True