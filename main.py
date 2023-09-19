import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random

# Загрузка файла XLSX
file_path = 'data.xlsx'
df = pd.read_excel(file_path)

random.seed(10)

try:
    # Преобразование столбца full_name
    full_name_part = df['full_name'].str.split(n=1, expand=True)[0].str.replace(' ', '').astype(int)
    df['full_name'] = df['full_name'].str.split(' ', 1).str[1]

    # Извлечение числового значения из столбца resale_price
    df['resale_price'] = df['resale_price'].str.extract('(\d+\.\d+)')
    df['resale_price'] = df['resale_price'].astype(float)  # Преобразование в числа (float)
    df['resale_price'] = [random.uniform(1.1, 17.2) for _ in range(len(df))]

    df['registered_year'] = full_name_part

    # Преобразование столбцов engine_capacity, kms_driven, max_power, mileage
    df['engine_capacity'] = df['engine_capacity'].str.extract('(\d+)').astype(float)
    # Рассчитываем среднее арифметическое значение столбца engine_capacity
    mean_engine_capacity = df['engine_capacity'].mean()
    # Заполняем пустые значения средним арифметическим, округленным до ближайшего целого
    df['engine_capacity'].fillna(round(mean_engine_capacity), inplace=True)

    df['insurance'].fillna('Third Party insurance',inplace=True)

    df['kms_driven'] = df['kms_driven'].str.replace(',', '').str.extract('(\d+)').astype(float)
    df['kms_driven'].fillna(random.randint(30000, 100000), inplace=True)

    df['max_power'] = df['max_power'].str.extract('(\d+\.\d+)').astype(float)
    # Рассчитываем среднее арифметическое значение столбца engine_capacity
    mean_max_power = df['max_power'].mean()
    # Заполняем пустые значения средним арифметическим, округленным до ближайшего целого
    df['max_power'].fillna(round(mean_max_power), inplace=True)

    mean_seats = df['seats'].mean()
    # Заполняем пустые значения средним арифметическим, округленным до ближайшего целого
    df['seats'].fillna(round(mean_seats), inplace=True)

    df['mileage'] = df['mileage'].str.extract('(\d+\.\d+)').astype(float)
    mean_mileage = df['mileage'].mean()
    df['mileage'].fillna(round(mean_mileage), inplace=True)



    # Преобразование столбца owner_type
    owner_mapping = {"First Owner": 1, "Second Owner": 2, "Third Owner": 3}
    df['owner_type'] = df['owner_type'].map(owner_mapping)
    df['owner_type'].fillna(random.randint(1,3),inplace=True)

    # Label Encoding для столбца Fuel_Type
    fuel_mapping = {"Petrol": 1, "Diesel": 2, "CNG": 3}
    df['fuel_type'] = df['fuel_type'].map(fuel_mapping)
    df['fuel_type'].fillna(random.randint(1,3),inplace=True)


    # Label Encoding для столбца transmission_type
    transmission_mapping = {"Manual": 0, "Automatic": 1}
    df['transmission_type'] = df['transmission_type'].map(transmission_mapping)

    # Label Encoding для столбца body_type
    body_type_mapping = {"SUV": 0, "Hatchback": 1, "Sedan": 2, "Toyota": 3, "Minivans": 4, "MUV": 5, "Cars": 6}
    df['body_type'] = df['body_type'].map(body_type_mapping)
    df['body_type'].fillna(random.randint(1, 6), inplace=True)

    # # Удаление строк с пустыми значениями
    # df.dropna(inplace=True)

    # Сохранение исправленных данных обратно в файл
    df.to_excel(file_path, index=False)

except Exception as e:
    print("Выполнить преобразование не удалось!")

# Исключение столбцов full_name и Insurance из признаков
X = df.drop(['resale_price', 'full_name', 'insurance','city'], axis=1)
y = df['resale_price']

# Разделите данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создайте модель линейной регрессии и обучите ее
model = LinearRegression()
model.fit(X_train, y_train)

# Оцените производительность модели
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Используйте модель для оценки стоимости автомобиля (пример входных данных)
input_data = [
    [15.0, 2017, 1435, 1, 50000, 1, 2, 83.1, 5, 15.5, 1]
]

predicted_price = model.predict(input_data)
print(f'Predicted Price: ${predicted_price[0]:.2f}')
