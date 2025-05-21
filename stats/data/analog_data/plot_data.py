import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from matplotlib.colors import TABLEAU_COLORS

def read_csv_with_metadata(filepath):
    """
    Читает CSV файл, пропуская метаданные в начале.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Находим вторую строку, начинающуюся на "Время," — это заголовок столбцов
    time_lines = [i for i, line in enumerate(lines) if line.startswith('Время,')]
    if len(time_lines) < 2:
        raise ValueError(f"В файле {filepath} не найдены необходимые строки 'Время,'.")
    header_idx = time_lines[1]
    
    # Заголовок и данные
    header_line = lines[header_idx].strip()
    data_lines = lines[header_idx + 1:]

    # Собираем текст для pandas
    csv_text = '\n'.join([header_line] + data_lines)
    return pd.read_csv(StringIO(csv_text), sep=',')

def group_columns_by_range(df):
    """
    Группирует столбцы по диапазонам значений.
    """
    ranges = {}
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        if column == 'Время':
            continue
        # Используем 95-й перцентиль вместо максимума для избежания выбросов
        col_max = np.abs(df[column]).quantile(0.95)
        
        if col_max < 1:
            ranges.setdefault('small', []).append((column, col_max))
        elif col_max < 100:
            ranges.setdefault('medium', []).append((column, col_max))
        else:
            ranges.setdefault('large', []).append((column, col_max))
    
    # Сортируем колонки внутри каждой группы по максимальному значению
    for range_key in ranges:
        ranges[range_key] = [col for col, _ in sorted(ranges[range_key], key=lambda x: x[1])]
    
    return ranges

def plot_csv_file(filepath):
    """
    Строит график для одного CSV файла и сохраняет его в отдельный файл.
    """
    try:
        df = read_csv_with_metadata(filepath)
        ranges = group_columns_by_range(df)
        
        # Создаем фигуру
        plt.figure(figsize=(34, 21), dpi=300)
        
        # Нормализуем время от 0 до 1
        time_normalized = np.linspace(0, 1, len(df['Время']))
        
        # Создаем список цветов из таблицы цветов matplotlib
        colors = list(TABLEAU_COLORS.values())
        
        # Порядок отображения графиков: большие значения сверху, малые снизу
        plot_order = ['large', 'medium', 'small']
        
        # Создаем оси с разными масштабами
        ax1 = plt.gca()  # Основная ось для больших значений
        ax2 = ax1.twinx()  # Вторая ось для средних значений
        ax3 = ax1.twinx()  # Третья ось для малых значений
        
        # Смещаем третью ось вправо
        ax3.spines['right'].set_position(('outward', 60))
        
        # Счетчик для цветов
        color_idx = 0
        
        # Рисуем графики для каждого диапазона
        for range_key in plot_order:
            if range_key not in ranges:
                continue
                
            columns = ranges[range_key]
            ax = ax1 if range_key == 'large' else (ax2 if range_key == 'medium' else ax3)
            
            for column in columns:
                color = colors[color_idx % len(colors)]
                color_idx += 1
                ax.plot(time_normalized, df[column], label=column, color=color, 
                       linewidth=1.5, alpha=0.8)
            
            
        
        # Настраиваем сетку и легенду
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Время', fontsize=24)
        
        # Объединяем легенды всех осей
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
                  bbox_to_anchor=(0.5, -0.15), loc='upper center', 
                  fontsize=24, ncol=2)
        
        # Заголовок
        plt.title(os.path.basename(filepath), fontsize=24, pad=20)
        
        # Настраиваем расстояние между графиками
        plt.tight_layout()
        
        # Создаем папку plots, если её нет
        plots_folder = 'plots'
        os.makedirs(plots_folder, exist_ok=True)
        
        # Сохраняем график с тем же именем, что и у CSV файла
        output_filename = os.path.splitext(os.path.basename(filepath))[0] + '.png'
        output_path = os.path.join(plots_folder, output_filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"График сохранен в файл: {output_path}")
        
    except Exception as e:
        print(f"Ошибка при обработке файла {filepath}: {e}")

def plot_all_csv_files(folder_path):
    """
    Строит графики для всех CSV файлов в папке.
    """
    # Получаем список CSV файлов
    csv_files = [f for f in os.listdir(folder_path) 
                if f.lower().endswith('.csv') and not f.startswith('_')]
    
    if not csv_files:
        print("CSV файлы не найдены")
        return
    
    # Обрабатываем каждый файл
    for fname in csv_files:
        filepath = os.path.join(folder_path, fname)
        plot_csv_file(filepath)

if __name__ == '__main__':
    folder = 'G:/4course/diploma/stats/data/analog_data'
    plot_all_csv_files(folder)
    print("Все графики сохранены в папке 'plots'")
