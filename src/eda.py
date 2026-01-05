import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


def read_data(train_path="../data/train.csv", test_path="../data/test.csv", sample_sub_path="../submissions/sample_submission.csv"):
    """Загружает train, test и sample_submission"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_sub_path)
    print("✅ Данные успешно загружены")
    return train, test, sample


def basic_info(df, name):
    """Выводит базовую информацию о датасете"""
    print(f"\n📊 Базовая информация: {name}")
    print(f"Размер: {df.shape}")
    print(f"Колонки: {list(df.columns)}")
    print(df.head())


def quality_checks(df, name, id_col="id"):
    """Проверяет пропуски, дубликаты и описательные статистики"""
    print(f"\n🔍 Проверка качества данных: {name}")
    miss = df.isna().sum().sort_values(ascending=False)
    print("Пропуски по колонкам:")
    print(miss[miss > 0])
    dup_rows = df.duplicated().sum()
    print(f"Дубликатов строк: {dup_rows}")
    if id_col in df.columns:
        dup_ids = df[id_col].duplicated().sum()
        print(f"Дубликатов id: {dup_ids}")


def plot_target_distribution(train):
    """Строит графики распределения целевой переменной"""
    if "y" not in train.columns:
        print("⚠️ Целевая переменная 'y' отсутствует")
        return
    counts = train["y"].value_counts()
    print("\n📈 Распределение целевой переменной:")
    print(counts)


def time_range_check(train, test):
    """Проверяет диапазон значений relative_date_number"""
    if "relative_date_number" not in train.columns or "relative_date_number" not in test.columns:
        print("⚠️ Колонка relative_date_number отсутствует")
        return
    print("\n⏳ Проверка диапазона времени:")
    print(f"Train: {train['relative_date_number'].min()} → {train['relative_date_number'].max()}")
    print(f"Test: {test['relative_date_number'].min()} → {test['relative_date_number'].max()}")


def correlation_analysis(train):
    """Считает корреляцию признаков с целевой переменной"""
    if "y" not in train.columns:
        print("⚠️ Целевая переменная 'y' отсутствует")
        return
    num_cols = [c for c in train.columns if c.startswith("x")]
    print("\n🔗 Корреляция признаков с y:")
    for c in num_cols:
        if train[c].nunique() > 1:
            rho = stats.spearmanr(train[c], train["y"]).correlation
            print(f"{c}: {rho:.3f}")


def time_aware_split(train, n_splits=5):
    """Создаёт временные сплиты для кросс-валидации"""
    if "relative_date_number" not in train.columns:
        print("⚠️ Нет relative_date_number для временного сплита")
        return []
    train_sorted = train.sort_values("relative_date_number").reset_index(drop=True)
    tss = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for tr_idx, val_idx in tss.split(train_sorted):
        splits.append((train_sorted.index[tr_idx].values, train_sorted.index[val_idx].values))
    print(f"\n📂 Создано {len(splits)} временных сплитов")
    return splits


def baseline_models(train, splits):
    """Запускает базовые модели и выводит F1"""
    if not splits or "y" not in train.columns:
        print("⚠️ Нет сплитов или целевой переменной")
        return
    features = [c for c in train.columns if c.startswith("x")] + ["g1", "g2", "relative_date_number"]
    X = train[features].copy()
    y = train["y"].values
    for col in ["g1", "g2"]:
        if col in X.columns and not np.issubdtype(X[col].dtype, np.number):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    X = X.fillna(X.median(numeric_only=True))

    for name, model in [
        ("RandomForest", RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced")),
        ("LightGBM", LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced"))
    ]:
        scores = []
        for tr_idx, val_idx in splits:
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            f1 = f1_score(y_val, preds)
            scores.append(f1)
        print(f"\n⚡ Базовая модель {name}: средний F1 = {np.mean(scores):.4f}")


def main():
    train, test, sample = read_data()
    basic_info(train, "train")
    basic_info(test, "test")
    basic_info(sample, "sample_submission")
    quality_checks(train, "train")
    quality_checks(test, "test")
    plot_target_distribution(train)
    time_range_check(train, test)
    correlation_analysis(train)
    splits = time_aware_split(train, n_splits=5)
    baseline_models(train, splits)
    print("\n✅ EDA завершён. Результаты сохранены в папке outputs")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Ошибка во время выполнения EDA: {e}", file=sys.stderr)
        sys.exit(1)
