import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Просмотр
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain head:")
print(train.head())

# Распределение y
print("\nTarget distribution:")
print(train['y'].value_counts(normalize=True))

# y
# 0.0    0.953607
# 1.0    0.046393
# Сильный дисбаланс классов → нужно использовать:
# class_weight='balanced' или is_unbalance=True в LightGBM/XGBoost.
# Метрику F1 как основную.
# Возможно: undersampling majority class или oversampling minority (SMOTE).


# Проверка пропусков
print("\nMissing values in train:")
print(train.isnull().sum())

# Основные статистики по x1-x12
print("\nNumerical features summary:")
print(train[[f'x{i}' for i in range(1,13)]].describe())

# проверка уникальных значений
# ... после загрузки данных

print("\nUnique values:")
print(f"g1: {train['g1'].nunique()} unique")
print(f"g2: {train['g2'].nunique()} unique")

print("\nTop g1:")
print(train['g1'].value_counts().head())

print("\nTop g2:")
print(train['g2'].value_counts().head())

# Агрегируем y по relative_date_number
time_trend = train.groupby('relative_date_number')['y'].agg(['mean', 'sum', 'count']).reset_index()
time_trend.columns = ['date', 'y_mean', 'y_sum', 'y_count']

plt.figure(figsize=(12, 6))
sns.lineplot(data=time_trend, x='date', y='y_mean', label='y_mean')
plt.title('Среднее значение y по времени')
plt.xlabel('relative_date_number')
plt.ylabel('Среднее y')
plt.grid(True)
plt.show()
# Среднее значение y по relative_date_number имеет синусоидальную форму с пиками около 5, 17 и 29 —
# это явный периодический паттерн.
# Это означает, что временная структура критически важна — вероятно, спрос зависит от циклических факторов.
