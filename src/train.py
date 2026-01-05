#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from features import prepare_datasets

def train_and_validate(train_feat: pd.DataFrame, n_splits=5):
    X = train_feat.drop(columns=["id", "y", "sample_weight"], errors="ignore")
    y = train_feat["y"]
    sample_weight = train_feat.get("sample_weight", None)

    tss = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(tss.split(X)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        w_tr = sample_weight.iloc[tr_idx] if sample_weight is not None else None

        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds)
        scores.append(f1)
        print(f"📂 Фолд {fold+1}: F1 = {f1:.4f}")

    print(f"\n✅ Средний F1 по {n_splits} фолдам: {np.mean(scores):.4f}")
    return model



def train_final_model(train_feat: pd.DataFrame):
    X = train_feat.drop(columns=["id", "y", "sample_weight"], errors="ignore")
    y = train_feat["y"]
    sample_weight = train_feat.get("sample_weight", None)

    model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=128,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X, y, sample_weight=sample_weight)
    print("✅ Финальная модель обучена")

    # --- Сохраняем важность признаков ---
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importances.to_csv("feature_importances.csv", index=False)
    print("📄 Важность признаков сохранена в feature_importances.csv")

    # Строим график топ-20 признаков
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importances.head(20), x="importance", y="feature", palette="viridis")
    plt.title("Топ-20 признаков по важности (LightGBM)")
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    print("📊 График важности признаков сохранён в feature_importances.png")

    return model



def make_submission(model, test_feat: pd.DataFrame, filename="my_submission.csv"):
    """
    Формирует файл my_submission.csv с предсказаниями.
    """
    # Убираем все служебные колонки
    X_test = test_feat.drop(columns=["id", "y", "sample_weight"], errors="ignore")

    # Проверка: совпадают ли признаки с обучением
    train_columns = model.feature_name_
    X_test = X_test.reindex(columns=train_columns, fill_value=0)

    preds = model.predict(X_test)
    submission = pd.DataFrame({
        "id": test_feat["id"],
        "y": preds.astype(int)
    })
    submission.to_csv(filename, index=False)
    print(f"📄 Файл {filename} сохранён")



if __name__ == "__main__":
    # Загружаем и формируем признаки
    train_feat, test_feat = prepare_datasets()

    # Валидация на временных сплитах
    model = train_and_validate(train_feat, n_splits=5)

    # Финальное обучение на всех данных
    final_model = train_final_model(train_feat)

    # Формируем my_submission.csv
    make_submission(final_model, test_feat)
