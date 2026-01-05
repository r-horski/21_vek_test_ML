import numpy as np
import pandas as pd

def create_features(df):
    """
    Генерация признаков: циклическое кодирование, агрегаты x1-x12.
    g1, g2 остаются в int для совместимости с time-aware encoding.
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


def add_time_aware_target_encoding(train_df, test_df, cols, target='y', alpha=10.0):
    """
    Time-aware Target Encoding: для каждой строки используем только прошлые значения.
    Предотвращает утечку данных из будущего.
    """
    train_out = train_df.copy()
    test_out = test_df.copy()

    global_mean = train_df[target].mean()

    # Сортируем трейн по времени
    train_out = train_out.sort_values(by='relative_date_number').reset_index(drop=True)

    for col in cols:
        # Накопленная сумма и счётчик до текущей строки (не включая её)
        cumsum = train_out.groupby(col)[target].cumsum() - train_out[target]
        cumcount = train_out.groupby(col).cumcount()

        # Сглаженное среднее: (sum + global_mean * alpha) / (count + alpha)
        smooth_mean = (cumsum + global_mean * alpha) / (cumcount + alpha)
        smooth_mean = smooth_mean.fillna(global_mean)  # для первой строки в группе

        train_out[f'{col}_tenc'] = smooth_mean

        # Для теста: используем всё train (уже безопасно, так как train до теста по времени)
        encodings = train_out.groupby(col)[target].agg(['mean', 'count'])
        encodings['smooth'] = (encodings['mean'] * encodings['count'] + global_mean * alpha) / (encodings['count'] + alpha)
        test_out[f'{col}_tenc'] = test_out[col].map(encodings['smooth']).fillna(global_mean)

    # Возвращаем в исходный порядок (если нужно)
    train_out = train_out.sort_values(by='index').reset_index(drop=True)

    return train_out, test_out