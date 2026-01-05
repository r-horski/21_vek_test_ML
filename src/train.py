import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

# Загрузка данных
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Генерация признаков
from features import create_features, add_time_aware_target_encoding

train_fe = create_features(train)
test_fe = create_features(test)

# Time-Aware Target Encoding (без утечки из будущего)
train_enc, test_enc = add_time_aware_target_encoding(
    train_df=train_fe,
    test_df=test_fe,
    cols=['g1', 'g2'],
    target='y',
    alpha=10.0
)

# Финальные признаки
features = [
    'relative_date_number',
    'date_sin', 'date_cos',
    'x_sum', 'x_nonzero'
] + [f'x{i}' for i in range(1, 13)] + \
  ['g1_tenc', 'g2_tenc']

cat_features = ['g1', 'g2']

# Подготовка признаков
X_train = train_enc[features + cat_features].copy()
y_train = train_enc['y']
X_test = test_enc[features + cat_features].copy()

# Преобразуем категориальные признаки
X_train[cat_features] = X_train[cat_features].astype('category')
X_test[cat_features] = X_test[cat_features].astype('category')

# Параметры модели
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
}

# Учёт дисбаланса через веса
neg_weight = len(y_train[y_train == 0]) / len(y_train)
pos_weight = len(y_train[y_train == 1]) / len(y_train)
params['scale_pos_weight'] = neg_weight / pos_weight

# Кросс-валидация
cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)  # Без перемешивания — учитываем временной порядок
preds = np.zeros(len(X_test))
f1_scores = []

for tr_idx, val_idx in cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    val_pred_proba = model.predict(X_val)
    val_pred = (val_pred_proba > 0.5).astype(int)
    f1 = f1_score(y_val, val_pred)
    f1_scores.append(f1)

print(f"Средний F1: {np.mean(f1_scores):.4f}")
print(f"Стандартное отклонение: {np.std(f1_scores):.4f}")

# Финальное предсказание на тесте
final_model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features),
    num_boost_round=int(1.1 * 50),  # немного больше, чем average best iteration
    callbacks=[lgb.log_evaluation(period=0)]
)

test_pred_proba = final_model.predict(X_test)
y_pred = (test_pred_proba > 0.5).astype(int)

# Сохранение результата
submission = pd.DataFrame({
    'id': test_enc['id'],
    'y': y_pred
})
submission.to_csv('../submissions/my_submission2.csv', index=False)

print("Файл my_submission2.csv сохранён.")
print(f"Распределение y в submission2:\n{submission['y'].value_counts().sort_index()}")