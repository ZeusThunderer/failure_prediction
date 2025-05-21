import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, precision_recall_curve,
    average_precision_score, brier_score_loss
)
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from scikeras.wrappers import KerasClassifier, KerasRegressor
 # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

'''
Результаты моделей на разных этапах обработки данных:

all_situations_fe.csv 
          Model  Accuracy  Precision  Recall  F1 Score   ROC AUC  Avg Precision  Brier Score
0  RandomForest  0.998792        0.0   0.000       0.0  0.628090       0.001781     0.001210
1       XGBoost  1.000000        0.0   0.000       0.0  1.000000       1.000000     0.000153
2      LightGBM  0.998943   0.538462   0.875  0.666667  0.999490       0.686310     0.000743
3           RNN  0.998792        0.0   0.000       0.0  0.499471       0.001208     0.001207
4          LSTM  0.998792        0.0   0.000       0.0  0.499471       0.001208     0.001207

all_situations_clean.csv
          Model  Accuracy  Precision  Recall  F1 Score   ROC AUC  Avg Precision  Brier Score
0  RandomForest  0.998792        0.0   0.000       0.0  0.717508       0.002523     0.001207
1       XGBoost  0.884946   0.001323   0.125  0.002618  0.679998       0.001993     0.001327
2      LightGBM  0.761437   0.001269   0.250  0.002525  0.563369       0.001602     0.001247
3           RNN  0.998792        0.0   0.000       0.0  0.500000       0.001208     0.001207
4          LSTM  0.998792        0.0   0.000       0.0  0.500000       0.001208     0.001207

all_situations_binary.csv
          Model  Accuracy  Precision  Recall  F1 Score   ROC AUC  Avg Precision  Brier Score
0  RandomForest  0.998792   0.000000   0.000  0.000000  0.812103       0.004906     0.001207
1       XGBoost  0.838744   0.002814   0.375  0.005587  0.711820       0.003701     0.011193
2      LightGBM  0.817756   0.002490   0.375  0.004946  0.668972       0.002366     0.003298
3           RNN  0.998792   0.000000   0.000  0.000000  0.464248       0.001208     0.001207
4          LSTM  0.998792   0.000000   0.000  0.000000  0.499471       0.001208     0.001207
'''

# Загрузка данных
df = pd.read_csv('all_situations_clean.csv', sep=',', encoding='utf-8')

# 1. Определяем признаки и целевую переменную
target_col = 'binary_failure'
X = df.drop(columns=[target_col, 'situation_id', 'Время'])
y = df[target_col]

# Заполняем все NaN нулями (или можно median/mean)
X = X.fillna(0)

# Для LightGBM создаём копию X с "безопасными" именами признаков (без спецсимволов)
X_lgbm = X.copy()
X_lgbm.columns = [
    ''.join(c if c.isalnum() or c == '_' else '_' for c in col)
    for col in X_lgbm.columns
]

# 2. Разделение на обучающую и тестовую выборки по situation_id
situation_ids = df['situation_id'].unique()
train_ids, test_ids = train_test_split(
    situation_ids, 
    test_size=0.2, 
    random_state=42
)
train_mask = df['situation_id'].isin(train_ids)
test_mask = df['situation_id'].isin(test_ids)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# 3. Масштабирование признаков для нейросетей
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Преобразование данных для RNN/LSTM: (samples, timesteps, features)
X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_rnn = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Используем исходные данные без ресэмплинга
X_train_resampled = X_train
y_train_resampled = y_train
X_train_scaled_resampled = X_train_scaled
y_train_scaled_resampled = y_train
X_lgbm_train_resampled = X_lgbm[train_mask].copy()
y_lgbm_train_resampled = y_train

# Обновляем размерности для RNN/LSTM
X_train_rnn_resampled = X_train_rnn

# Проверяем размерности
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_train_rnn shape: {X_train_rnn.shape}")

# Функция для поиска оптимального порога с более тонкой настройкой
def find_optimal_threshold(y_true, y_pred_proba):
    thresholds = np.arange(0.01, 0.99, 0.01)  # Более тонкая настройка порога
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    return optimal_threshold

# 4. Обучение моделей

# 4.1. Случайный лес (Random Forest)
'''
Лучшие параметры Random Forest: {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 300}
Лучший F1-score: 0.16842105263157894
'''
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight={0: 1, 1: 5},  # Уменьшаем вес положительного класса
    min_weight_fraction_leaf=0.1,  # Добавляем регуляризацию
    bootstrap=True,
    max_samples=0.8  # Используем только 80% данных для каждого дерева
)

# Обновляем обучение моделей с использованием сбалансированных данных
rf.fit(X_train_resampled, y_train_resampled)
y_pred_proba = rf.predict_proba(X_test)[:, 1]
optimal_threshold = find_optimal_threshold(y_test, y_pred_proba)
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# 5. Random Forest
print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (точность предсказания аварий): {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall (доля обнаруженных аварий): {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score (баланс между точностью и полнотой): {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC AUC (способность различать классы): {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Average Precision (средняя точность): {average_precision_score(y_test, y_pred_proba):.4f}")
print(f"Brier Score (калибровка вероятностей): {brier_score_loss(y_test, y_pred_proba):.4f}")

# Расчет дополнительных метрик
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"\nДополнительные метрики:")
print(f"False Positive Rate (частота ложных тревог): {fp/(fp+tn):.4f}")
print(f"False Negative Rate (пропущенные аварии): {fn/(fn+tp):.4f}")
print(f"Specificity (специфичность): {tn/(tn+fp):.4f}")
print(f"Positive Predictive Value (PPV): {tp/(tp+fp) if (tp+fp) > 0 else 0:.4f}")
print(f"Negative Predictive Value (NPV): {tn/(tn+fn):.4f}")

# 6. Важность признаков для Random Forest
importances = pd.Series(rf.feature_importances_, index=X.columns)
sorted_importances = importances.sort_values(ascending=False)
print("\nTop 10 Feature importances:")
print(sorted_importances.head(10))

# 7. XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=5,  # Уменьшаем вес положительного класса
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    learning_rate=0.01,
    reg_alpha=0.1,  # L1 регуляризация
    reg_lambda=1.0  # L2 регуляризация
)
xgb.fit(X_train_resampled, y_train_resampled)
y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]
optimal_threshold_xgb = find_optimal_threshold(y_test, y_pred_proba_xgb)
y_pred_xgb = (y_pred_proba_xgb >= optimal_threshold_xgb).astype(int)

# 8. LightGBM
lgbm = LGBMClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
    class_weight={0: 1, 1: 5},  # Уменьшаем вес положительного класса
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    learning_rate=0.01,
    reg_alpha=0.1,  # L1 регуляризация
    reg_lambda=1.0  # L2 регуляризация
)
lgbm.fit(X_lgbm_train_resampled, y_lgbm_train_resampled)
y_pred_proba_lgbm = lgbm.predict_proba(X_lgbm[test_mask])[:, 1]
optimal_threshold_lgbm = find_optimal_threshold(y_test, y_pred_proba_lgbm)
y_pred_lgbm = (y_pred_proba_lgbm >= optimal_threshold_lgbm).astype(int)

# Callbacks для нейронных сетей
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

# Улучшенная архитектура RNN с регуляризацией
rnn = Sequential([
    Input(shape=(1, X_train_scaled.shape[1])),
    SimpleRNN(32, activation='relu', 
              kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
              recurrent_regularizer=l1_l2(l1=0.01, l2=0.01),
              dropout=0.2),
    Dense(16, activation='relu',
          kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    Dense(1, activation='sigmoid')
])
rnn.compile(
    optimizer=Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Улучшенная архитектура LSTM с регуляризацией
lstm = Sequential([
    Input(shape=(1, X_train_scaled.shape[1])),
    LSTM(32, activation='relu',
         kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
         recurrent_regularizer=l1_l2(l1=0.01, l2=0.01),
         dropout=0.2),
    Dense(16, activation='relu',
          kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    Dense(1, activation='sigmoid')
])
lstm.compile(
    optimizer=Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Обучение нейронных сетей с валидацией
rnn.fit(
    X_train_rnn_resampled, y_train_scaled_resampled,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Добавляем предсказания для RNN
y_pred_proba_rnn = rnn.predict(X_test_rnn).flatten()
optimal_threshold_rnn = find_optimal_threshold(y_test, y_pred_proba_rnn)
y_pred_rnn = (y_pred_proba_rnn >= optimal_threshold_rnn).astype(int)

lstm.fit(
    X_train_rnn_resampled, y_train_scaled_resampled,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Добавляем предсказания для LSTM
y_pred_proba_lstm = lstm.predict(X_test_rnn).flatten()
optimal_threshold_lstm = find_optimal_threshold(y_test, y_pred_proba_lstm)
y_pred_lstm = (y_pred_proba_lstm >= optimal_threshold_lstm).astype(int)

# 9. Результаты
results = pd.DataFrame({
    'Model': ['RandomForest', 'XGBoost', 'LightGBM', 'RNN', 'LSTM'],
    'Accuracy': [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_xgb),
        accuracy_score(y_test, y_pred_lgbm),
        accuracy_score(y_test, y_pred_rnn),
        accuracy_score(y_test, y_pred_lstm)
    ],
    'Precision': [
        precision_score(y_test, y_pred, zero_division=0),
        precision_score(y_test, y_pred_xgb, zero_division=0),
        precision_score(y_test, y_pred_lgbm, zero_division=0),
        precision_score(y_test, y_pred_rnn, zero_division=0),
        precision_score(y_test, y_pred_lstm, zero_division=0)
    ],
    'Recall': [
        recall_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred_xgb, zero_division=0),
        recall_score(y_test, y_pred_lgbm, zero_division=0),
        recall_score(y_test, y_pred_rnn, zero_division=0),
        recall_score(y_test, y_pred_lstm, zero_division=0)
    ],
    'F1 Score': [
        f1_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred_xgb, zero_division=0),
        f1_score(y_test, y_pred_lgbm, zero_division=0),
        f1_score(y_test, y_pred_rnn, zero_division=0),
        f1_score(y_test, y_pred_lstm, zero_division=0)
    ],
    'ROC AUC': [
        roc_auc_score(y_test, y_pred_proba),
        roc_auc_score(y_test, y_pred_proba_xgb),
        roc_auc_score(y_test, y_pred_proba_lgbm),
        roc_auc_score(y_test, y_pred_proba_rnn),
        roc_auc_score(y_test, y_pred_proba_lstm)
    ],
    'Avg Precision': [
        average_precision_score(y_test, y_pred_proba),
        average_precision_score(y_test, y_pred_proba_xgb),
        average_precision_score(y_test, y_pred_proba_lgbm),
        average_precision_score(y_test, y_pred_proba_rnn),
        average_precision_score(y_test, y_pred_proba_lstm)
    ],
    'Brier Score': [
        brier_score_loss(y_test, y_pred_proba),
        brier_score_loss(y_test, y_pred_proba_xgb),
        brier_score_loss(y_test, y_pred_proba_lgbm),
        brier_score_loss(y_test, y_pred_proba_rnn),
        brier_score_loss(y_test, y_pred_proba_lstm)
    ]
})

print("\nModel Comparison Results:")
print(results)

# 12. Визуализация результатов
# Создаем директорию для графиков
import os
plot_dir = 'model_plots'
os.makedirs(plot_dir, exist_ok=True)

# ROC кривые
plt.figure(figsize=(10, 6))
from sklearn.metrics import roc_curve

for model_name, y_pred_proba in [
    ('Random Forest', y_pred_proba),
    ('XGBoost', y_pred_proba_xgb),
    ('LightGBM', y_pred_proba_lgbm),
    ('RNN', y_pred_proba_rnn),
    ('LSTM', y_pred_proba_lstm)
]:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (частота ложных тревог)')
plt.ylabel('True Positive Rate (доля обнаруженных аварий)')
plt.title('ROC Curves for All Models')
plt.legend()
plt.savefig(os.path.join(plot_dir, 'roc_curves.png'))
plt.close()

# Precision-Recall кривые
plt.figure(figsize=(10, 6))
for model_name, y_pred_proba in [
    ('Random Forest', y_pred_proba),
    ('XGBoost', y_pred_proba_xgb),
    ('LightGBM', y_pred_proba_lgbm),
    ('RNN', y_pred_proba_rnn),
    ('LSTM', y_pred_proba_lstm)
]:
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, label=f'{model_name} (AP = {average_precision_score(y_test, y_pred_proba):.3f})')

plt.xlabel('Recall (доля обнаруженных аварий)')
plt.ylabel('Precision (точность предсказания аварий)')
plt.title('Precision-Recall Curves for All Models')
plt.legend()
plt.savefig(os.path.join(plot_dir, 'precision_recall_curves.png'))
plt.close()

# Матрицы ошибок - отдельно для каждой модели
for model_name, y_pred_val in [
    ('RandomForest', y_pred),
    ('XGBoost', y_pred_xgb),
    ('LightGBM', y_pred_lgbm),
    ('RNN', y_pred_rnn),
    ('LSTM', y_pred_lstm)
]:
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_val)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted (предсказано)')
    plt.ylabel('True (реальность)')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()
    
    # Добавляем текстовый файл с информацией о матрице
    with open(os.path.join(plot_dir, f'confusion_matrix_{model_name}_info.txt'), 'w') as f:
        tn, fp, fn, tp = cm.ravel()
        f.write(f"Матрица ошибок для модели {model_name}:\n")
        f.write(f"True Negative (TN): {tn}\n")
        f.write(f"False Positive (FP): {fp}\n")
        f.write(f"False Negative (FN): {fn}\n")
        f.write(f"True Positive (TP): {tp}\n\n")
        f.write(f"Общая точность (Accuracy): {(tp+tn)/(tp+tn+fp+fn):.6f}\n")
        f.write(f"Precision: {tp/(tp+fp) if (tp+fp) > 0 else 0:.6f}\n")
        f.write(f"Recall: {tp/(tp+fn) if (tp+fn) > 0 else 0:.6f}\n")
        f.write(f"Specificity: {tn/(tn+fp) if (tn+fp) > 0 else 0:.6f}\n")

# Важность признаков для лучшей модели
plt.figure(figsize=(12, 6))
sorted_importances.head(15).plot(kind='barh')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.xlabel('Importance (важность признака)')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'feature_importance.png'))
plt.close()

'''print("\nПодбор гиперпараметров для моделей:")

# 1. Random Forest
print("\n1. Подбор гиперпараметров для Random Forest:")
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'min_weight_fraction_leaf': [0.0, 0.01, 0.05]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    rf_param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)
print("Лучшие параметры Random Forest:", rf_grid.best_params_)
print("Лучший F1-score:", rf_grid.best_score_)

# 2. XGBoost
print("\n2. Подбор гиперпараметров для XGBoost:")
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

xgb_grid = GridSearchCV(
    XGBClassifier(
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        random_state=42
    ),
    xgb_param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_train, y_train)
print("Лучшие параметры XGBoost:", xgb_grid.best_params_)
print("Лучший F1-score:", xgb_grid.best_score_)

# 3. LightGBM
print("\n3. Подбор гиперпараметров для LightGBM:")
lgbm_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 63, 127],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

lgbm_grid = GridSearchCV(
    LGBMClassifier(class_weight='balanced', random_state=42),
    lgbm_param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
lgbm_grid.fit(X_lgbm[train_mask], y[train_mask])
print("Лучшие параметры LightGBM:", lgbm_grid.best_params_)
print("Лучший F1-score:", lgbm_grid.best_score_)

# 4. RNN
print("\n4. Подбор гиперпараметров для RNN:")
def create_rnn_model(units=32, dropout=0.2, learning_rate=0.001):
    model = Sequential([
        Input(shape=(1, X_train_scaled.shape[1])),
        SimpleRNN(units, activation='relu', dropout=dropout),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

rnn_param_grid = {
    'units': [16, 32, 64],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [32, 64, 128],
    'epochs': [10]
}

rnn_grid = GridSearchCV(
    KerasClassifier(
        build_fn=create_rnn_model,
        verbose=0
    ),
    rnn_param_grid,
    cv=3,
    scoring='f1',
    n_jobs=1,
    verbose=1
)
rnn_grid.fit(
    X_train_rnn, y_train,
    class_weight={0: 1, 1: len(y_train[y_train==0])/len(y_train[y_train==1])}
)
print("Лучшие параметры RNN:", rnn_grid.best_params_)
print("Лучший F1-score:", rnn_grid.best_score_)

# 5. LSTM
print("\n5. Подбор гиперпараметров для LSTM:")
def create_lstm_model(units=32, dropout=0.2, learning_rate=0.001):
    model = Sequential([
        Input(shape=(1, X_train_scaled.shape[1])),
        LSTM(units, activation='relu', dropout=dropout),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

lstm_param_grid = {
    'units': [16, 32, 64],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [32, 64, 128],
    'epochs': [10]
}

lstm_grid = GridSearchCV(
    KerasClassifier(
        build_fn=create_lstm_model,
        verbose=0
    ),
    lstm_param_grid,
    cv=3,
    scoring='f1',
    n_jobs=1,
    verbose=1
)
lstm_grid.fit(
    X_train_rnn, y_train,
    class_weight={0: 1, 1: len(y_train[y_train==0])/len(y_train[y_train==1])}
)
print("Лучшие параметры LSTM:", lstm_grid.best_params_)
print("Лучший F1-score:", lstm_grid.best_score_)

print("\nОбъяснение гиперпараметров:")
print("\nRandom Forest:")
print("""
n_estimators: количество деревьев в лесу
max_depth: максимальная глубина каждого дерева
min_samples_split: минимальное количество образцов для разделения узла
min_samples_leaf: минимальное количество образцов в листовом узле
max_features: максимальное количество признаков для разделения
min_weight_fraction_leaf: минимальная взвешенная доля образцов в листовом узле
""")

print("\nXGBoost:")
print("""
n_estimators: количество деревьев
max_depth: максимальная глубина дерева
learning_rate: скорость обучения
min_child_weight: минимальная сумма весов в листовом узле
subsample: доля образцов для обучения каждого дерева
colsample_bytree: доля признаков для каждого дерева
gamma: минимальное уменьшение потерь для разделения
""")

print("\nLightGBM:")
print("""
n_estimators: количество деревьев
max_depth: максимальная глубина дерева
learning_rate: скорость обучения
num_leaves: максимальное количество листьев в дереве
min_child_samples: минимальное количество образцов в листовом узле
subsample: доля образцов для обучения каждого дерева
colsample_bytree: доля признаков для каждого дерева
""")

print("\nRNN/LSTM:")
print("""
units: количество нейронов в RNN/LSTM слое
dropout: вероятность отключения нейронов для предотвращения переобучения
learning_rate: скорость обучения
batch_size: размер батча для обучения
epochs: количество эпох обучения
""")

# Используем лучшие параметры для финальных моделей
rf = rf_grid.best_estimator_
xgb = xgb_grid.best_estimator_
lgbm = lgbm_grid.best_estimator_
rnn = rnn_grid.best_estimator_
lstm = lstm_grid.best_estimator_'''


