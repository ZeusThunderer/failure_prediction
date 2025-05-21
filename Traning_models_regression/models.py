import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import os

'''
Результаты моделей на разных этапах обработки данных:

all_situations_fe.csv 
          Model       MAE       MSE      RMSE        R2
0  RandomForest  0.035922  0.001878  0.043332  0.914129
1       XGBoost  0.085850  0.012371  0.111225  0.434249
2      LightGBM  0.087142  0.014287  0.119530  0.346607
3           RNN  0.057095  0.009166  0.095738  0.579311
4          LSTM  0.053834  0.004393  0.066276  0.798393

all_situations_clean.csv
          Model       MAE       MSE      RMSE        R2 time regular
0  RandomForest  0.036811  0.002036  0.045117  0.955594
1       XGBoost  0.051316  0.003154  0.056159  0.931199
2      LightGBM  0.040506  0.002184  0.046731  0.952361
3           RNN  0.160763  0.029512  0.171791  0.356184
4          LSTM  0.079534  0.007175  0.084706  0.843471

          Model       MAE       MSE      RMSE        R2
0  RandomForest  0.044570  0.003512  0.059266  0.839370
1       XGBoost  0.098806  0.017482  0.132221  0.200487
2      LightGBM  0.084971  0.015002  0.122484  0.313910
3           RNN  0.065582  0.009695  0.098462  0.555033
4          LSTM  0.091485  0.016308  0.127703  0.251501
_all_situations_with_warnings.csv
          Model       MAE        MSE      RMSE           R2
0  RandomForest  0.053969   0.003752  0.061250     0.828432
1       XGBoost  0.075492   0.009375  0.096825     0.571261
2      LightGBM  0.050844   0.004725  0.068737     0.783928
3           RNN  1.010222   4.540051  2.130739  -207.377300
4          LSTM  3.199594  40.102637  6.332664 -1839.613624
'''

# all_situations.csv (сырые), all_situations_clean.csv (очищенные), all_situations_fe.csv (с новыми признаками)
df = pd.read_csv('all_situations_fe.csv', sep=',', encoding='utf-8')

# 1. Определяем признаки и целевую переменную
target_col = 'prob'
X = df.drop(columns=[target_col, 'situation_id'])  # Удаляем situation_id из признаков
y = df[target_col]

# Заполняем все NaN нулями (или можно median/mean)
X = X.fillna(X.mean())

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

# 4. Преобразование данных для RNN/LSTM
# Группируем данные по situation_id и создаем последовательности
def create_sequences(data, target, situation_ids, df_filtered, sequence_length=10):
    sequences = []
    targets = []
    
    for situation_id in situation_ids:
        # Используем маску для фильтрации данных
        mask = df_filtered['situation_id'] == situation_id
        situation_data = data[mask]
        situation_target = target[mask]
        
        if len(situation_data) >= sequence_length:
            for i in range(len(situation_data) - sequence_length + 1):
                sequences.append(situation_data[i:i + sequence_length])
                targets.append(situation_target.iloc[i + sequence_length - 1])
    
    return np.array(sequences), np.array(targets)

# Создаем последовательности для обучения и тестирования
sequence_length = 10  # Длина последовательности
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, train_ids, df[train_mask], sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, test_ids, df[test_mask], sequence_length)

# Теперь X_train_seq имеет форму (n_samples, sequence_length, n_features)

# 4. Обучение моделей
# 4.1. Случайный лес (Random Forest)
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 5. Random Forest
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# 6. Важность признаков для Random Forest
importances = pd.Series(rf.feature_importances_, index=X.columns)
sorted_importances = importances.sort_values(ascending=False)
print("Feature importances:")
print(sorted_importances.head(10))

# 7. XGBoost
xgb = XGBRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# 8. LightGBM
lgbm = LGBMRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
lgbm.fit(X_lgbm[train_mask], y[train_mask])
y_pred_lgbm = lgbm.predict(X_lgbm[test_mask])

# 9. RNN
rnn = Sequential([
    SimpleRNN(32, input_shape=(1, X_train_scaled.shape[1]), activation='relu'),
    Dense(1)
])
rnn.compile(optimizer=Adam(0.001), loss='mse')
rnn.fit(X_train_seq, y_train_seq, epochs=10, batch_size=64, verbose=1)
y_pred_rnn = rnn.predict(X_test_seq).flatten()

# 10. LSTM
lstm = Sequential([
    LSTM(32, input_shape=(1, X_train_scaled.shape[1]), activation='relu'),
    Dense(1)
])
lstm.compile(optimizer=Adam(0.001), loss='mse')
lstm.fit(X_train_seq, y_train_seq, epochs=10, batch_size=64, verbose=1)
y_pred_lstm = lstm.predict(X_test_seq).flatten()


# 12. Результаты
results = pd.DataFrame({
    'Model': ['RandomForest', 'XGBoost', 'LightGBM', 'RNN', 'LSTM'],
    'MAE': [
        mean_absolute_error(y_test, y_pred),
        mean_absolute_error(y_test, y_pred_xgb),
        mean_absolute_error(y_test, y_pred_lgbm),
        mean_absolute_error(y_test_seq, y_pred_rnn),
        mean_absolute_error(y_test_seq, y_pred_lstm)
    ],
    'MSE': [
        mean_squared_error(y_test, y_pred),
        mean_squared_error(y_test, y_pred_xgb),
        mean_squared_error(y_test, y_pred_lgbm),
        mean_squared_error(y_test_seq, y_pred_rnn),
        mean_squared_error(y_test_seq, y_pred_lstm)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test, y_pred)),
        np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        np.sqrt(mean_squared_error(y_test, y_pred_lgbm)),
        np.sqrt(mean_squared_error(y_test_seq, y_pred_rnn)),
        np.sqrt(mean_squared_error(y_test_seq, y_pred_lstm))
    ],
    'R2': [
        r2_score(y_test, y_pred),
        r2_score(y_test, y_pred_xgb),
        r2_score(y_test, y_pred_lgbm),
        r2_score(y_test_seq, y_pred_rnn),
        r2_score(y_test_seq, y_pred_lstm)
    ]
})

print(results)

# Визуализация результатов регрессии

# Создаем директорию для сохранения визуализаций
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Создаем поддиректории для каждой модели
model_names = ['RandomForest', 'XGBoost', 'LightGBM', 'RNN', 'LSTM']
for model_name in model_names:
    if not os.path.exists(f'visualizations/{model_name}'):
        os.makedirs(f'visualizations/{model_name}')

# 1. Визуализация важности признаков для Random Forest
plt.figure(figsize=(12, 8))
sorted_importances.head(15).plot(kind='barh')
plt.title('Важность признаков (Random Forest)')
plt.tight_layout()
plt.savefig('visualizations/RandomForest/feature_importance.png')
plt.close()

# Подготовка данных для визуализации
models_data = [
    ('RandomForest', y_test, y_pred),
    ('XGBoost', y_test, y_pred_xgb),
    ('LightGBM', y_test, y_pred_lgbm),
    ('RNN', y_test_seq, y_pred_rnn),
    ('LSTM', y_test_seq, y_pred_lstm)
]

# 2. Фактические vs Предсказанные значения (отдельно для каждой модели)
for model_name, y_true, y_predicted in models_data:
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_predicted, alpha=0.5)
    
    # Добавляем линию идеального предсказания
    min_val = min(y_true.min(), y_predicted.min())
    max_val = max(y_true.max(), y_predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name}/actual_vs_predicted.png')
    plt.close()

# 3. Визуализация остатков (Residuals) - отдельно для каждой модели
for model_name, y_true, y_predicted in models_data:
    plt.figure(figsize=(10, 8))
    residuals = y_true - y_predicted
    plt.scatter(y_predicted, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'{model_name}: Residuals')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Остатки')
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name}/residuals.png')
    plt.close()

# 4. Распределение ошибок - отдельно для каждой модели
for model_name, y_true, y_predicted in models_data:
    plt.figure(figsize=(10, 8))
    residuals = y_true - y_predicted
    sns.histplot(residuals, kde=True)
    plt.title(f'{model_name}: Distribution of Errors')
    plt.xlabel('Ошибка')
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name}/error_distribution.png')
    plt.close()

# 5. Сравнение метрик между моделями
# MAE
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MAE', data=results)
plt.title('Сравнение моделей по MAE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/model_comparison_MAE.png')
plt.close()

# MSE
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MSE', data=results)
plt.title('Сравнение моделей по MSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/model_comparison_MSE.png')
plt.close()

# RMSE
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', data=results)
plt.title('Сравнение моделей по RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/model_comparison_RMSE.png')
plt.close()

# R2 score
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2', data=results)
plt.title('Сравнение моделей по R2 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/model_comparison_R2.png')
plt.close()

# 6. Сохранение метрик каждой модели в отдельный файл
for i, model_name in enumerate(model_names):
    # Создадим текстовый отчет по каждой модели
    with open(f'visualizations/{model_name}/metrics.txt', 'w') as f:
        f.write(f"Метрики модели {model_name}:\n")
        f.write(f"MAE: {results.iloc[i]['MAE']:.6f}\n")
        f.write(f"MSE: {results.iloc[i]['MSE']:.6f}\n")
        f.write(f"RMSE: {results.iloc[i]['RMSE']:.6f}\n")
        f.write(f"R2: {results.iloc[i]['R2']:.6f}\n")

print("Визуализации для каждой модели сохранены в отдельных папках в директории 'visualizations/'")

