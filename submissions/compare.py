import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Стиль графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Загрузка файлов
submission = pd.read_csv('../submissions/my_submission.csv')
sample = pd.read_csv('../submissions/sample_submission.csv')
test_data = pd.read_csv('../data/test.csv')

print("ПРОВЕРКА ФОРМАТА ПРЕДСКАЗАНИЙ")
print(f"Количество строк в моём файле: {len(submission)}")
print(f"Количество строк в sample: {len(sample)}")

if len(submission) == len(sample):
    print("Количество строк: OK")
else:
    print("Количество строк не совпадает!")

print(f"Имена колонок: {list(submission.columns)}")
if list(submission.columns) == ['id', 'y']:
    print("Колонки: OK")
else:
    print("Неверные названия колонок")

if submission['id'].equals(sample['id']):
    print("ID полностью совпадают")
else:
    print("ID не совпадают — возможна ошибка порядка")

print(f"Уникальные значения в y: {submission['y'].unique()}")
if set(submission['y'].unique()) <= {0, 1}:
    print("Все значения y корректны")
else:
    print("Есть недопустимые значения в y")

missing = submission.isnull().sum().sum()
print(f"Пропущенных значений: {missing}")
if missing == 0:
    print("Пропусков нет")

print("\nРаспределение классов:")
print(submission['y'].value_counts().sort_index())

# === Визуализация ===
fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=120)

# 1. Сравнение долей классов
mine = submission['y'].value_counts().sort_index() / len(submission)
sample_y = sample['y'].value_counts().sort_index() / len(sample)

labels = ['Класс 0', 'Класс 1']
width = 0.35

axes[0].bar(np.arange(2) - width/2, mine.values, width, label='Мой submission', color='C0', alpha=0.8)
axes[0].bar(np.arange(2) + width/2, sample_y.values, width, label='Образец', color='C1', alpha=0.8)
axes[0].set_title('Сравнение долей целевого класса')
axes[0].set_ylabel('Доля')
axes[0].set_xticks(range(2))
axes[0].set_xticklabels(labels)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# 2. Поведение y по времени
test_with_pred = test_data[['id', 'relative_date_number']].merge(submission, on='id')
time_trend_mine = test_with_pred.groupby('relative_date_number')['y'].mean()

test_with_sample = test_data[['id', 'relative_date_number']].merge(sample, on='id')
time_trend_sample = test_with_sample.groupby('relative_date_number')['y'].mean()

axes[1].plot(time_trend_mine.index, time_trend_mine.values,
             label='Мой прогноз', color='blue', linewidth=2)
axes[1].plot(time_trend_sample.index, time_trend_sample.values,
             label='Образец', color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1].set_title('Среднее y по времени (relative_date_number)')
axes[1].set_xlabel('relative_date_number')
axes[1].set_ylabel('Среднее значение y')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()