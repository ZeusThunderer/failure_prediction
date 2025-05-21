import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score

# Этот скрипт выполняет первичный анализ (EDA) датасета аварийных ситуаций.

# Путь к файлу с исходными данными
DATA_PATH = os.path.join(os.path.dirname(__file__), '_all_situations_with_warnings.csv')

# Колонки, которые нужно исключить из анализа
EXCLUDE_COLS = ['situation_id', 'Время', 'prob', 'general_alarm']

# Создаём папки для визуализаций
vis_dir = os.path.join(os.path.dirname(__file__), 'eda_visualizations')
os.makedirs(vis_dir, exist_ok=True)

def format_feature_name(name):
    """Форматирует название признака для лучшей читаемости на графиках"""
    # Удаляем общие префиксы
    name = name.replace('Сигнал_', '')
    name = name.replace('Параметр_', '')
    
    # Сокращаем длинные названия
    if len(name) > 30:
        parts = name.split('_')
        if len(parts) > 2:
            # Оставляем только первые буквы каждого слова, кроме последнего
            shortened = ''.join(p[0] for p in parts[:-1]) + '_' + parts[-1]
            return shortened[:30]
    return name

# Функция для создания тепловой карты корреляций
def plot_correlation_heatmap(df, prefix=''):
    # Создаем фигуру с дополнительным пространством снизу для легенды
    plt.figure(figsize=(20, 20))
    corr_matrix = df.drop(columns=EXCLUDE_COLS).corr()
    
    # Создаем словарь для соответствия номеров и названий признаков
    feature_names = {i: format_feature_name(col) for i, col in enumerate(corr_matrix.columns)}
    
    # Переименовываем колонки и индексы в порядковые номера
    corr_matrix = corr_matrix.rename(columns={col: str(i) for i, col in enumerate(corr_matrix.columns)},
                                   index={col: str(i) for i, col in enumerate(corr_matrix.columns)})
    
    # Создаем сетку для размещения тепловой карты и легенды
    gs = plt.GridSpec(2, 1, height_ratios=[4, 1])
    ax1 = plt.subplot(gs[0])
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax1)
    ax1.set_title(f'{prefix}Корреляционная матрица', fontsize=14)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=14)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=14)
    
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{prefix}correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Функция для анализа информативности признаков
def analyze_feature_importance(df, target_col='prob'):
    numeric_cols = df.drop(columns=EXCLUDE_COLS).select_dtypes(include=[np.number]).columns
    importance_metrics = {}
    
    for col in numeric_cols:
        if col != target_col:
            # Корреляция с целевой переменной
            correlation = df[col].corr(df[target_col])
            
            # Взаимная информация
            mi_score = mutual_info_score(df[col].fillna(df[col].mean()), 
                                       df[target_col].fillna(df[target_col].mean()))
            
            # Дисперсия
            variance = df[col].var()
            
            importance_metrics[col] = {
                'correlation': correlation,
                'mutual_info': mi_score,
                'variance': variance
            }
    
    # Создаем DataFrame с метриками
    importance_df = pd.DataFrame(importance_metrics).T
    
    # Форматируем названия признаков
    importance_df.index = [format_feature_name(col) for col in importance_df.index]
    
    # Визуализация важности признаков
    plt.figure(figsize=(20, 6))
    
    plt.subplot(1, 3, 1)
    sns.barplot(x=importance_df.index, y='correlation', data=importance_df)
    plt.title('Корреляция с целевой переменной')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.subplot(1, 3, 2)
    sns.barplot(x=importance_df.index, y='mutual_info', data=importance_df)
    plt.title('Взаимная информация')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.subplot(1, 3, 3)
    sns.barplot(x=importance_df.index, y='variance', data=importance_df)
    plt.title('Дисперсия')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df

print('='*50)
print('АНАЛИЗ ИСХОДНОГО ДАТАСЕТА')
print('='*50)

# Загрузка данных
print('Загрузка данных...')
df = pd.read_csv(DATA_PATH)

# Общая информация о датасете
print('\n1. ОБЩАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ:')
print('-'*30)
print(f'Количество строк: {df.shape[0]:,}')
print(f'Количество столбцов: {df.shape[1]}')
print(f'Количество уникальных ситуаций: {df["situation_id"].nunique()}')
print(f'Размер датасета в памяти: {df.memory_usage().sum() / 1024**2:.2f} MB')

# Информация о пропусках (исключая специальные колонки)
print('\n2. АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:')
print('-'*30)
missing = df.drop(columns=EXCLUDE_COLS).isnull().mean() * 100
print(f'Среднее количество пропусков: {missing.mean():.2f}%')
print(f'Медиана пропусков: {missing.median():.2f}%')
print(f'Минимум пропусков: {missing.min():.2f}%')
print(f'Максимум пропусков: {missing.max():.2f}%')
print('\nСтолбцы с пропусками (>0%):')
print(missing[missing > 0].sort_values(ascending=False))

# Статистика по числовым признакам (исключая специальные колонки)
print('\n3. СТАТИСТИКА ПО ЧИСЛОВЫМ ПРИЗНАКАМ:')
print('-'*30)
numeric_stats = df.drop(columns=EXCLUDE_COLS).describe()
print(numeric_stats)

# Удаление столбцов с более чем 70% пропусков (исключая специальные колонки)
threshold = 0.7
target_col = 'prob'
cols_to_drop = [col for col in missing[missing/100 > threshold].index if col not in EXCLUDE_COLS]
if cols_to_drop:
    print(f'\nУдаляем {len(cols_to_drop)} столбцов с >70% пропусков: {cols_to_drop}')
    df = df.drop(columns=cols_to_drop)
else:
    print('\nНет столбцов с >70% пропусков (кроме исключенных).')

# Обработка мультиколлинеарности
print('\nОбработка мультиколлинеарности:')
print('-'*30)
corr_matrix = df.drop(columns=EXCLUDE_COLS).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
if to_drop:
    print(f'Удаляем признаки с корреляцией > 0.95: {to_drop}')
    df = df.drop(columns=to_drop)
else:
    print('Нет признаков с корреляцией > 0.95')

# Время от 0 до n секунд в рамках каждой ситуации
df = df.sort_values(['situation_id', 'Время'])
df['Время'] = df.groupby('situation_id').cumcount()

print('\n='*50)
print('АНАЛИЗ ОЧИЩЕННОГО ДАТАСЕТА')
print('='*50)

# Общая информация после очистки
print('\n1. ОБЩАЯ ИНФОРМАЦИЯ ПОСЛЕ ОЧИСТКИ:')
print('-'*30)
print(f'Количество строк: {df.shape[0]:,}')
print(f'Количество столбцов: {df.shape[1]}')
print(f'Количество уникальных ситуаций: {df["situation_id"].nunique()}')
print(f'Размер датасета в памяти: {df.memory_usage().sum() / 1024**2:.2f} MB')

# Информация о пропусках после очистки (исключая специальные колонки)
print('\n2. АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ПОСЛЕ ОЧИСТКИ:')
print('-'*30)
missing_after = df.drop(columns=EXCLUDE_COLS).isnull().mean() * 100
print(f'Среднее количество пропусков: {missing_after.mean():.2f}%')
print(f'Медиана пропусков: {missing_after.median():.2f}%')
print(f'Минимум пропусков: {missing_after.min():.2f}%')
print(f'Максимум пропусков: {missing_after.max():.2f}%')
print('\nСтолбцы с пропусками (>0%):')
print(missing_after[missing_after > 0].sort_values(ascending=False))

# Статистика по числовым признакам после очистки (исключая специальные колонки)
print('\n3. СТАТИСТИКА ПО ЧИСЛОВЫМ ПРИЗНАКАМ ПОСЛЕ ОЧИСТКИ:')
print('-'*30)
numeric_stats_after = df.drop(columns=EXCLUDE_COLS).describe().T
print(numeric_stats_after)

# Создание визуализаций
print('\nСоздание визуализаций...')

# Корреляционные матрицы
plot_correlation_heatmap(pd.read_csv(DATA_PATH), prefix='raw_')
plot_correlation_heatmap(df, prefix='clean_')

# Анализ важности признаков
importance_df = analyze_feature_importance(df)

# Сохранение результатов анализа важности признаков
importance_file = os.path.join(os.path.dirname(__file__), 'feature_importance_analysis.txt')
with open(importance_file, 'w', encoding='utf-8') as f:
    f.write('АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ\n')
    f.write('='*50 + '\n\n')
    
    # Топ-10 признаков по корреляции
    f.write('Топ-10 признаков по корреляции с целевой переменной:\n')
    f.write('-'*30 + '\n')
    f.write(importance_df.sort_values('correlation', ascending=False).head(10).to_string())
    f.write('\n\n')
    
    # Топ-10 признаков по взаимной информации
    f.write('Топ-10 признаков по взаимной информации:\n')
    f.write('-'*30 + '\n')
    f.write(importance_df.sort_values('mutual_info', ascending=False).head(10).to_string())
    f.write('\n\n')
    
    # Топ-10 признаков по дисперсии
    f.write('Топ-10 признаков по дисперсии:\n')
    f.write('-'*30 + '\n')
    f.write(importance_df.sort_values('variance', ascending=False).head(10).to_string())

print(f'\nАнализ важности признаков сохранен в файл: {importance_file}')
print(f'Все визуализации сохранены в директории: {vis_dir}')

# Сохранение статистики в файл
stats_file = os.path.join(os.path.dirname(__file__), 'dataset_statistics.txt')
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write('СТАТИСТИКА ДАТАСЕТА АВАРИЙНЫХ СИТУАЦИЙ\n')
    f.write('='*50 + '\n\n')
    
    f.write('ИСХОДНЫЙ ДАТАСЕТ:\n')
    f.write('-'*30 + '\n')
    f.write(f'Размер: {df.shape[0]:,} строк × {df.shape[1]} столбцов\n')
    f.write(f'Уникальных ситуаций: {df["situation_id"].nunique()}\n')
    f.write(f'Пропуски: {missing.mean():.2f}% (среднее), {missing.median():.2f}% (медиана)\n\n')
    
    f.write('ОЧИЩЕННЫЙ ДАТАСЕТ:\n')
    f.write('-'*30 + '\n')
    f.write(f'Размер: {df.shape[0]:,} строк × {df.shape[1]} столбцов\n')
    f.write(f'Уникальных ситуаций: {df["situation_id"].nunique()}\n')
    f.write(f'Пропуски: {missing_after.mean():.2f}% (среднее), {missing_after.median():.2f}% (медиана)\n\n')
    
    f.write('СТАТИСТИКА ПО ПРИЗНАКАМ:\n')
    f.write('-'*30 + '\n')
    f.write(numeric_stats_after.to_string())

print(f'\nПодробная статистика сохранена в файл: {stats_file}')

# Сохраняем очищенный датасет
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'all_situations_clean.csv')
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nОчищенный датасет сохранён в: {OUTPUT_PATH}")

