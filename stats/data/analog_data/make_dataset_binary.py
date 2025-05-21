import os
import pandas as pd
from io import StringIO


def read_situation_csv(path, situation_id, round_time=0):
    """
    Считывает один CSV, пропуская метаданные,
    объединяет дубли по времени (берёт первые ненулевые значения) и добавляет столбец situation_id.
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

    # Добавляем столбец ситуации
    df.insert(0, 'situation_id', situation_id)
    
    return df


def consolidate_folder(folder_path, round_time=0):
    """
    Обходит все CSV в папке, читает и объединяет их в один DataFrame.
    При этом отсутствующие столбцы заполняются NaN.
    """
    all_dfs = []
    csv_files = sorted([fname for fname in os.listdir(folder_path) if fname.lower().endswith('.csv')])
    i = 1
    for fname in csv_files:
        if fname.startswith('_'):
            continue
        fullpath = os.path.join(folder_path, fname)
        try:
            df = read_situation_csv(fullpath, situation_id=i, round_time=round_time)
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

    # Читаем binary_failure из существующего файла
    try:
        existing_binary = pd.read_csv('_all_situations_binary.csv')
        # Создаем ключ для объединения из situation_id и Время
        result['merge_key'] = result['situation_id'].astype(str) + '_' + result['Время'].astype(str)
        existing_binary['merge_key'] = existing_binary['situation_id'].astype(str) + '_' + existing_binary['Время'].astype(str)
        
        # Получаем столбец binary_failure
        binary_failure = existing_binary.set_index('merge_key')['binary_failure']
        result = result.set_index('merge_key')
        result['binary_failure'] = binary_failure
        result = result.reset_index(drop=True)
        
        print("Добавлен столбец binary_failure из существующего файла")
    except Exception as e:
        print(f"Не удалось прочитать binary_failure из файла: {e}")
        print("Создаем столбец binary_failure со значениями 0")
        result['binary_failure'] = 0

    # Сохраняем объединённый файл
    result.to_csv('_all_situations_binary.csv', index=False, encoding='utf-8')
