import numpy as np
from sklearn.model_selection import KFold

def create_features(df):
    """
    Генерация признаков: циклическое кодирование, агрегаты x1-x12.
    g1, g2 остаются в исходном типе (int) для корректной работы Target Encoding.
    """
    df = df.copy()

    # Циклическое кодирование с периодом 12 (по паттерну)
    period = 12
    df['date_sin'] = np.sin(2 * np.pi * df['relative_date_number'] / period)
    df['date_cos'] = np.cos(2 * np.pi * df['relative_date_number'] / period)

    # Суммарные признаки
    x_cols = [f'x{i}' for i in range(1, 13)]
    df['x_sum'] = df[x_cols].sum(axis=1)
    df['x_nonzero'] = (df[x_cols] > 0).sum(axis=1)

    return df


def add_target_encoding(train_df, test_df, cols, target='y', alpha=10.0):
    """
    Сглаженное Target Encoding с кросс-валидацией на train, применение к test.
    alpha — сила сглаживания.
    """
    train_out = train_df.copy()
    test_out = test_df.copy()

    global_mean = train_df[target].mean()

    # Обработка теста
    for col in cols:
        encodings = train_df.groupby(col)[target].agg(['mean', 'count'])
        encodings['smooth'] = (
            encodings['mean'] * encodings['count'] + global_mean * alpha
        ) / (encodings['count'] + alpha)
        test_out[f'{col}_tenc'] = test_out[col].map(encodings['smooth']).fillna(global_mean)

    # Обработка трейна (out-of-fold)
    for col in cols:
        train_out[f'{col}_tenc'] = 0.0
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(train_df):
            tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
            encodings = tr.groupby(col)[target].agg(['mean', 'count'])
            encodings['smooth'] = (
                encodings['mean'] * encodings['count'] + global_mean * alpha
            ) / (encodings['count'] + alpha)
            val_encoded = val[col].map(encodings['smooth']).fillna(global_mean)
            train_out.loc[val_idx, f'{col}_tenc'] = val_encoded

    return train_out, test_out