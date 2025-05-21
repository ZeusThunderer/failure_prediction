import pandas as pd
import os

# Этот скрипт выполняет генерацию новых признаков (feature engineering) для временных рядов аварийных ситуаций.
# Для каждого числового признака по каждой ситуации рассчитываются скользящие статистики, разности и индикатор пропуска.

# Путь к исходному датасету
DATA_PATH = os.path.join(os.path.dirname(__file__), 'all_situations_clean.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'all_situations_fe.csv')

print('Загрузка данных...')
df = pd.read_csv(DATA_PATH)

# Определяем признаки, для которых будут создаваться новые признаки
# Исключаем целевые переменные и служебные столбцы (situation_id, Время)
target_cols = ['prob', 'general_alarm']  # prob — индекс тревожности, general_alarm — бинарный индикатор аварийного состояния
service_cols = ['situation_id', 'Время']
features = [col for col in df.columns if col not in target_cols + service_cols]

# Длина окна для скользящих статистик (10 секунд)
window = 10
print(f'Генерируем rolling/diff признаки с окном {window} секунд...')
for col in features:
    # Скользящее среднее по окну 10 секунд (по каждой ситуации отдельно)
    df[f'{col}_rollmean_{window}'] = df.groupby('situation_id')[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    # Скользящее стандартное отклонение по окну 10 секунд
    df[f'{col}_rollstd_{window}'] = df.groupby('situation_id')[col].transform(lambda x: x.rolling(window=window, min_periods=1).std())
    # Первая разность (насколько быстро меняется сигнал)
    df[f'{col}_diff1'] = df.groupby('situation_id')[col].diff()
    # Вторая разность (ускорение изменений)
    df[f'{col}_diff2'] = df.groupby('situation_id')[col].diff().diff()
    # Индикатор пропуска (1 — значение отсутствует, 0 — есть)
    df[f'{col}_isnull'] = df[col].isnull().astype(int)

print('Сохраняем результат...')
df.to_csv(OUTPUT_PATH, index=False)
print(f'Готово! Файл с новыми признаками: {OUTPUT_PATH}') 