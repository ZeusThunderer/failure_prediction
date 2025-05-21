import os
import pandas as pd
import json

# Папка с CSV-файлами
csv_folder = 'G:/4course/diploma/stats/data/analog_data'  # Замените на путь
ustavki_path = 'ustavki.json'

# Загружаем уставки
with open(ustavki_path, 'r', encoding='utf-8') as f:
    thresholds = json.load(f)

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Найдём, где заголовок
    header_index = next(i for i, line in enumerate(lines) if line.strip().startswith('Время,"'))

    # Метаинфо — сохраняем как есть
    metadata = ''.join(lines[:header_index])

    # Чтение данных
    df = pd.read_csv(filepath, skiprows=header_index)
    
    # Округляем время до целого
    df['Время'] = df['Время'].round()
    
    # Удаляем дубликаты, оставляя строки с большим количеством непустых значений
    df['non_empty_count'] = df.count(axis=1)
    df = df.sort_values('non_empty_count', ascending=False)
    df = df.drop_duplicates(subset=['Время'], keep='first')
    df = df.drop('non_empty_count', axis=1)
    
    # Сортируем по времени по возрастанию
    df = df.sort_values('Время')

    # Вычисляем prob и general_alarm
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

    # Сохраняем результат
    output_path = os.path.join(csv_folder, os.path.basename(filepath))
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write(metadata)
    df.to_csv(output_path, mode='a', index=False)

# Обрабатываем все CSV
for filename in os.listdir(csv_folder):
    if filename.lower().endswith('.csv'):
        process_file(os.path.join(csv_folder, filename))

print("Файлы обновлены: добавлен столбец prob, general_alarm теперь бинарный (1 при превышении 90% уставки).")
