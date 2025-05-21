import os

def convert_csv_encoding(root_dir):
    """
    Рекурсивно конвертирует все CSV-файлы из CP1251 в UTF-8
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                
                try:
                    # Читаем содержимое в CP1251
                    with open(file_path, 'r', encoding='cp1251') as f:
                        content = f.read()
                    
                    # Перезаписываем файл в UTF-8
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"Успешно конвертирован: {file_path}")
                
                except UnicodeDecodeError:
                    print(f"Файл не в CP1251: {file_path}")
                except Exception as e:
                    print(f"Ошибка обработки файла {file_path}: {str(e)}")

if __name__ == "__main__":
    convert_csv_encoding('./data')