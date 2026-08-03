"""Carga, auditoria y preparacion sin fuga de datos de California Housing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42
TARGET = "MedHouseVal"


@dataclass
class DataBundle:
    """Splits originales y estandarizados junto con sus transformadores."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    X_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    X_test_scaled: np.ndarray
    scaler: StandardScaler


def load_housing(data_home: str | Path = "data/sklearn"):
    """Descarga/cachea el dataset y lo devuelve como Bunch con DataFrame."""

    Path(data_home).mkdir(parents=True, exist_ok=True)
    return fetch_california_housing(data_home=data_home, as_frame=True)


def quality_report(frame: pd.DataFrame) -> dict:
    """Resume integridad y atipicos mediante la regla IQR de 1.5."""

    q1 = frame.quantile(0.25)
    q3 = frame.quantile(0.75)
    iqr = q3 - q1
    outliers = ((frame < (q1 - 1.5 * iqr)) | (frame > (q3 + 1.5 * iqr))).sum()
    target_max = float(frame[TARGET].max())
    return {
        "shape": tuple(frame.shape),
        "nulls": int(frame.isna().sum().sum()),
        "duplicates": int(frame.duplicated().sum()),
        "iqr_outliers": {key: int(value) for key, value in outliers.items()},
        "target_min": float(frame[TARGET].min()),
        "target_max": target_max,
        "target_at_cap": int((frame[TARGET] >= target_max).sum()),
    }


def split_and_scale(frame: pd.DataFrame, seed: int = SEED) -> DataBundle:
    """Divide 70/15/15 y ajusta StandardScaler solo con entrenamiento."""

    X = frame.drop(columns=TARGET)
    y = frame[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, shuffle=True
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    return DataBundle(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        X_test_scaled=X_test_scaled,
        scaler=scaler,
    )


def validate_bundle(bundle: DataBundle) -> None:
    """Falla temprano si el split o el escalado violan los invariantes."""

    assert len(bundle.X_train) == 14_448
    assert len(bundle.X_val) == 3_096
    assert len(bundle.X_test) == 3_096
    assert bundle.X_train.shape[1] == 8
    assert np.allclose(bundle.X_train_scaled.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(bundle.X_train_scaled.std(axis=0), 1.0, atol=1e-5)
    train_ids = set(bundle.X_train.index)
    assert train_ids.isdisjoint(bundle.X_val.index)
    assert train_ids.isdisjoint(bundle.X_test.index)
    assert set(bundle.X_val.index).isdisjoint(bundle.X_test.index)

