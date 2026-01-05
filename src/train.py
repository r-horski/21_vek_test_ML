import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

# Загрузка
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Генерация признаков
from features import create_features, add_target_encoding

train_fe = create_features(train)
test_fe = create_features(test)

# Target Encoding
train_enc, test_enc = add_target_encoding(
    train_fe, test_fe,
    cols=['g1', 'g2'],
    target='y',
    alpha=10.0  # сглаживание — можно подбирать
)

# Финальные признаки
features = [
    'relative_date_number',
    'date_sin', 'date_cos',
    'x_sum', 'x_nonzero'
] + [f'x{i}' for i in range(1, 13)] + \
  ['g1_tenc', 'g2_tenc']

cat_features = ['g1', 'g2']  # будем использовать как категориальные в LGBM

X_train = train_enc[features + cat_features].copy()
y_train = train_enc['y']
X_test = test_enc[features + cat_features].copy()

# Переводим категориальные признаки в тип 'category' ДО обучения
X_train[cat_features] = X_train[cat_features].astype('category')
X_test[cat_features] = X_test[cat_features].astype('category')

# LightGBM с балансировкой
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'is_unbalance': True
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros(len(X_test))
f1_scores = []

for tr_idx, val_idx in cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    # Создание датасетов LGBM
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    # Обучение
    model = lgb.train(
    params,
    dtrain,
    valid_sets=[dtrain, dval],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100)  # скрываем логи
    ]
)

    # Предсказания на валидации
    val_pred_proba = model.predict(X_val)
    val_pred = (val_pred_proba > 0.5).astype(int)  # порог 0.5
    f1 = f1_score(y_val, val_pred)
    f1_scores.append(f1)
    print(f"F1 на фолде: {f1:.4f}")

    # Предсказания на тесте
    preds += model.predict(X_test) / cv.n_splits

print(f"Средний F1 на кросс-валидации: {np.mean(f1_scores):.4f}")
print(f"Стандартное отклонение F1: {np.std(f1_scores):.4f}")

# Преобразуем вероятности в бинарные предсказания: порог 0.5
y_pred = (preds > 0.5).astype(int)

# Создаём submission
submission = pd.DataFrame({
    'id': test_enc['id'],
    'y': y_pred
})

# Сохраняем
submission.to_csv('../submissions/my_submission.csv', index=False)

print("Файл my_submission.csv успешно сохранён!")
print(f"Распределение предсказаний:\n{pd.Series(y_pred).value_counts()}")