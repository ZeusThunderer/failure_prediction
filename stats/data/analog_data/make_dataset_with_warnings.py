import os
import pandas as pd
import json
from io import StringIO


def load_thresholds(ustavki_path='ustavki.json'):
    """Загружает уставки из JSON файла"""
    with open(ustavki_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_warnings(df, thresholds):
    """Вычисляет prob и general_alarm для DataFrame"""
    prob_values = []
    general_alarm = []
    
    for _, row in df.iterrows():
        scores = []
        alarm_triggered = False
        for col, rules in thresholds.items():
            if col not in df.columns:
                continue
            try:
                value = float(row[col])
                if 'ВА' in rules:
                    proximity = value / rules['ВА']
                    proximity = max(0.0, min(proximity, 1.0))
                    scores.append(proximity)
                    # Проверяем превышение 90% от уставки
                    if proximity >= 0.9:
                        alarm_triggered = True
            except:
                continue
        prob_values.append(sum(scores)/len(scores) if scores else 0.0)
        general_alarm.append(1 if alarm_triggered else 0)
    
    df['prob'] = prob_values
    df['general_alarm'] = general_alarm
    return df


def read_situation_csv(path, situation_id, thresholds, round_time=0):
    """
    Считывает один CSV, пропуская метаданные,
    объединяет дубли по времени (берёт первые ненулевые значения) и добавляет столбец situation_id.
    Также вычисляет prob и general_alarm.
    """
    # Читаем все строки
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    
    # Находим вторую строку, начинающуюся на "Время," — это заголовок столбцов
    time_lines = [i for i, line in enumerate(lines) if line.startswith('Время,')]
    if len(time_lines) < 2:
        raise ValueError(f"В файле {path} не найдены необходимые строки 'Время,'.")
    header_idx = time_lines[1]
    
    # Заголовок и данные
    header_line = lines[header_idx].strip()
    data_lines = lines[header_idx + 1:]

    # Собираем текст для pandas
    csv_text = '\n'.join([header_line] + data_lines)
    df = pd.read_csv(StringIO(csv_text), sep=',', dtype=float)

    # Округляем время
    if 'Время' in df.columns:
        df['Время'] = df['Время'].round(round_time)

    # Объединяем строки с одинаковым временем: берём первое ненулевое значение в группе
    def first_non_null(series):
        non_null = series.dropna()
        return non_null.iloc[0] if not non_null.empty else float('nan')

    df = (
        df.groupby('Время', as_index=False)
          .agg(first_non_null)
    )

    # Сортируем по времени
    df = df.sort_values('Время')

    # Вычисляем prob и general_alarm
    df = calculate_warnings(df, thresholds)

    # Добавляем столбец ситуации
    df.insert(0, 'situation_id', situation_id)
    return df


def consolidate_folder(folder_path, round_time=0):
    """
    Обходит все CSV в папке, читает и объединяет их в один DataFrame.
    При этом отсутствующие столбцы заполняются NaN.
    """
    # Загружаем уставки
    thresholds = load_thresholds()

    all_dfs = []
    csv_files = sorted([fname for fname in os.listdir(folder_path) if fname.lower().endswith('.csv')])
    i = 1
    for fname in csv_files:
        if fname.startswith('_'):
            continue
        fullpath = os.path.join(folder_path, fname)
        try:
            df = read_situation_csv(fullpath, situation_id=i, thresholds=thresholds, round_time=round_time)
            all_dfs.append(df)
            print(f"  + {fname}: {df.shape[0]} строк, {df.shape[1]} столбцов")
            i += 1
        except Exception as e:
            print(f"Ошибка в файле {fname}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    # Объединяем все DataFrame, отсутствующие колонки станут NaN
    combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    return combined


if __name__ == '__main__':
    folder = 'G:/4course/diploma/stats/data/analog_data'
    result = consolidate_folder(folder)
    print("Итого строк:", result.shape[0])
    print("Итого столбцов:", result.shape[1])
    # Сохраняем объединённый файл
    result.to_csv('_all_situations_with_warnings.csv', index=False, encoding='utf-8')