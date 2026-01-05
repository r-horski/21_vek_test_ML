#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def make_features(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    """
    Функция подготовки признаков для модели.
    Вход: датафрейм train или test.
    Выход: датафрейм с новыми признаками.
    """

    # --- Кодирование категориальных признаков ---
    for col in ["g1", "g2"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # --- Временные признаки ---
    # Нормализованная дата (от 0 до 1)
    df["date_norm"] = df["relative_date_number"] / df["relative_date_number"].max()

    # Индикаторы начала и конца периода
    df["is_start_period"] = (df["relative_date_number"] <= 5).astype(int)
    df["is_end_period"] = (df["relative_date_number"] >= df["relative_date_number"].max() - 5).astype(int)

    # --- Инженерия признаков ---
    # Взаимодействия между сильными признаками
    df["x4_x12"] = df["x4"] * df["x12"]
    df["x10_x12"] = df["x10"] * df["x12"]

    # Индикаторы нулевых значений
    for col in [f"x{i}" for i in range(1, 13)]:
        df[f"{col}_is_zero"] = (df[col] == 0).astype(int)

    # Логарифмирование для признаков с длинным хвостом
    for col in ["x4", "x5", "x10", "x12"]:
        df[f"{col}_log1p"] = np.log1p(df[col])

    # --- Агрегаты по группам ---
    # Среднее значение признаков внутри g1/g2
    for col in ["g1", "g2"]:
        for feat in [f"x{i}" for i in range(1, 13)]:
            df[f"{feat}_mean_by_{col}"] = df.groupby(col)[feat].transform("mean")

    # --- Балансировка классов (только для train) ---
    if is_train and "y" in df.columns:
        # Вес для каждого объекта: больше для редкого класса
        df["sample_weight"] = df["y"].apply(lambda v: 5 if v == 1 else 1)

    return df


def prepare_datasets(train_path="../data/train.csv", test_path="../data/test.csv"):
    """
    Загружает train и test, формирует признаки и возвращает готовые датафреймы.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_feat = make_features(train, is_train=True)
    test_feat = make_features(test, is_train=False)

    print("✅ Признаки успешно сформированы")
    print(f"Train shape: {train_feat.shape}, Test shape: {test_feat.shape}")

    return train_feat, test_feat


if __name__ == "__main__":
    train_feat, test_feat = prepare_datasets()
