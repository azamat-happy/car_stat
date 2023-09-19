import pandas as pd
import numpy as np

# Загрузка файла XLSX
file_path = 'data.xlsx'
df = pd.read_excel(file_path)

try:
    # Преобразование столбца full_name
    df['full_name'] = df['full_name'].str.split(' ', 1).str[1]

    # Извлечение числового значения из столбца resale_price
    df['resale_price'] = df['resale_price'].str.extract('(\d+\.\d+)')
    df['resale_price'] = df['resale_price'].astype(float)  # Преобразование в числа (float)

    # Преобразование столбцов engine_capacity, kms_driven, max_power, mileage
    df['engine_capacity'] = df['engine_capacity'].str.extract('(\d+)').astype(float)
    df['kms_driven'] = df['kms_driven'].str.replace(',', '').str.extract('(\d+)').astype(float)
    df['max_power'] = df['max_power'].str.extract('(\d+\.\d+)').astype(float)
    df['mileage'] = df['mileage'].str.extract('(\d+\.\d+)').astype(float)

    # Преобразование столбца owner_type
    owner_mapping = {"First Owner": 1, "Second Owner": 2, "Third Owner": 3}
    df['owner_type'] = df['owner_type'].map(owner_mapping)

    # Удаление строк с пустыми значениями
    df.dropna(inplace=True)

    # Сохранение исправленных данных обратно в файл
    df.to_excel(file_path, index=False)
except Exception as e:
    print("Выполнить преобразование не удалось!")
