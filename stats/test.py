input_file = "./data/2024-10-16_08-57-48/ГПА31_Журнал_Событий_01042023.csv"
output_file = "./data/2024-10-16_08-57-48/filtered_discrete.csv"

# Условия фильтрации
allowed_types = {"Предупреждение", "Авария"}

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    header_passed = False
    for line in infile:
        if not header_passed:
            outfile.write(line)
            if line.strip().startswith("Дата,Время"):
                header_passed = True
            continue

        # Обрабатываем только строки после заголовка таблицы
        
        if any(allowed_type in line for allowed_type in allowed_types):
            outfile.write(line)
