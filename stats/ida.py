import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path):
    """
    Функция читает CSV-файл с учетом того, что в файле:
      - первые 8 строк – метаданные,
      - 9-я строка – пустая (или разделитель),
      - 10-я строка – заголовок столбцов,
      - 11-я строка – строка с диапазонами значений (пропускаем её),
      - с 12-й строки – сами данные.
    """
    skiprows = list(range(9)) + [10]
    df = pd.read_csv(file_path, skiprows=skiprows)
    df['Время'] = pd.to_numeric(df['Время'], errors='coerce')
    for col in df.columns:
        if col != 'Время':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def normalize_series(series):
    """
    Нормализует серию методом min-max, переводя значения в диапазон [0, 1].
    Если все значения одинаковые, возвращает исходную серию.
    """
    if series.max() == series.min():
        return series
    return (series - series.min()) / (series.max() - series.min())

def plot_all_columns_normalized(df, file_path, output_dir="plots"):
    """
    Строит один график, на котором отображаются нормализованные значения всех столбцов
    (кроме 'Время') в зависимости от времени. Нормализация проводится методом min-max.
    График сохраняется в папку output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(36, 24))
    for col in df.columns:
        if col == 'Время':
            continue
        #norm_series = normalize_series(df[col])
        plt.plot(df['Время'], df[col], linestyle='-', label=col)
    plt.xlabel("Время")
    plt.ylabel("Нормализованное значение")
    plt.title(f"{os.path.basename(file_path)}: нормализованные измерения")
    plt.legend()
    output_file = os.path.join(output_dir, f"{os.path.basename(file_path)}.png")
    plt.savefig(output_file)
    plt.close()

def calculate_metrics(df):
    """
    Рассчитывает базовые метрики для каждого столбца (кроме 'Время'):
      - среднее,
      - стандартное отклонение,
      - минимум,
      - максимум,
      - число пропущенных значений.
    """
    metrics = {}
    for col in df.columns:
        if col == 'Время':
            continue
        series = df[col]
        metrics[col] = {
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": int(series.isna().sum())
        }
    return metrics

def perform_eda(df):
    """
    Проводит первичную разведку данных:
      - описательная статистика,
      - корреляционная матрица.
    """
    eda_results = {}
    eda_results['describe'] = df.describe().to_dict()
    eda_results['correlation'] = df.corr().to_dict()
    return eda_results

def process_file(file_path):
    """
    Обрабатывает один файл: читает данные, строит нормализованный график,
    рассчитывает метрики и проводит EDA.
    Возвращает словарь с рассчитанными метриками и результатами EDA.
    """
    print(f"Обработка файла: {file_path}")
    df = read_data(file_path)
    plot_all_columns_normalized(df, file_path)
    metrics = calculate_metrics(df)
    eda = perform_eda(df)
    return {"metrics": metrics, "eda": eda}

def main():
    base_path = './data'
    pattern = os.path.join(base_path, '*', '*_аналоговые*.csv')
    files = glob.glob(pattern)
    results = {}
    for file in files:
        results[file] = process_file(file)
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("Анализ завершен. Результаты сохранены в results.json")

if __name__ == '__main__':
    main()