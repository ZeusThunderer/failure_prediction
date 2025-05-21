import os
import pandas as pd

def convert_xlsx_to_csv(root_dir):
    """
    Рекурсивно конвертирует все XLSX-файлы в CSV в указанной директории и её поддиректориях
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.xlsx'):
                # Формируем полные пути
                xlsx_path = os.path.join(dirpath, filename)
                csv_filename = filename.rsplit('.', 1)[0] + '.csv'
                csv_path = os.path.join(dirpath, csv_filename)
                
                try:
                    # Читаем XLSX файл
                    df = pd.read_excel(xlsx_path)
                    
                    # Сохраняем в CSV
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    print(f"Конвертирован: {xlsx_path} -> {csv_path}")
                    
                except Exception as e:
                    print(f"Ошибка при конвертации {xlsx_path}: {str(e)}")

if __name__ == "__main__":
    # Укажите корневую директорию для обработки
    convert_xlsx_to_csv('./data')