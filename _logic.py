import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Для масштабирования данных перед передачей их в модель линейной регрессии
from sklearn.preprocessing import StandardScaler
import random


def _logic_MainWindow(input_data):
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
        # df['resale_price'] = [random.uniform(1.1, 17.2) for _ in range(len(df))]

        df['registered_year'] = full_name_part

        # Преобразование столбцов engine_capacity, kms_driven, max_power, mileage
        df['engine_capacity'] = df['engine_capacity'].str.extract('(\d+)').astype(float)
        # Рассчитываем среднее арифметическое значение столбца engine_capacity
        mean_engine_capacity = df['engine_capacity'].mean()
        # Заполняем пустые значения средним арифметическим, округленным до ближайшего целого
        # df['engine_capacity'].fillna(round(mean_engine_capacity), inplace=True)

        # df['insurance'].fillna('Third Party insurance', inplace=True)

        df['kms_driven'] = df['kms_driven'].str.replace(',', '').str.extract('(\d+)').astype(float)
        # df['kms_driven'].fillna(random.randint(30000, 100000), inplace=True)

        df['max_power'] = df['max_power'].str.extract('(\d+\.\d+)').astype(float)
        # Рассчитываем среднее арифметическое значение столбца engine_capacity
        mean_max_power = df['max_power'].mean()
        # Заполняем пустые значения средним арифметическим, округленным до ближайшего целого
        # df['max_power'].fillna(round(mean_max_power), inplace=True)

        mean_seats = df['seats'].mean()
        # Заполняем пустые значения средним арифметическим, округленным до ближайшего целого
        # df['seats'].fillna(round(mean_seats), inplace=True)

        df['mileage'] = df['mileage'].str.extract('(\d+\.\d+)').astype(float)
        mean_mileage = df['mileage'].mean()
        # df['mileage'].fillna(round(mean_mileage), inplace=True)

        # Преобразование столбца owner_type
        owner_mapping = {"First Owner": 1, "Second Owner": 2, "Third Owner": 3}
        df['owner_type'] = df['owner_type'].map(owner_mapping)
        # df['owner_type'].fillna(random.randint(1, 3), inplace=True)

        # Label Encoding для столбца Fuel_Type
        fuel_mapping = {"Petrol": 1, "Diesel": 2, "CNG": 3}
        df['fuel_type'] = df['fuel_type'].map(fuel_mapping)
        # df['fuel_type'].fillna(random.randint(1, 3), inplace=True)

        # Label Encoding для столбца transmission_type
        transmission_mapping = {"Manual": 0, "Automatic": 1}
        df['transmission_type'] = df['transmission_type'].map(transmission_mapping)

        # Label Encoding для столбца body_type
        body_type_mapping = {"SUV": 0, "Hatchback": 1, "Sedan": 2, "Toyota": 3, "Minivans": 4, "MUV": 5, "Cars": 6}
        df['body_type'] = df['body_type'].map(body_type_mapping)
        # df['body_type'].fillna(random.randint(1, 6), inplace=True)

        # # Удаление строк с пустыми значениями
        df.dropna(inplace=True)

        # Сохранение исправленных данных обратно в файл
        df.to_excel(file_path, index=False)

    except Exception as e:
        print("Выполнить преобразование не удалось!")

    # Исключение столбцов full_name и Insurance из признаков
    X = df.drop(['resale_price', 'full_name', 'insurance', 'city', 'Column1', 'mileage'], axis=1)
    y = df['resale_price']

    # Разделите данные на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создайте модель линейной регрессии и обучите ее
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Оцените производительность модели
    y_pred = model.predict(X_test)

    # Вычисление коэффициента детерминации (R^2)
    r2 = r2_score(y_test, y_pred)
    print(f'Коэффициент детерминации (R^2): {r2}')

    # вычисление среднеквадратичной ошибки
    mse = mean_squared_error(y_test, y_pred)
    print(f'Ошибка аппроксимации (MSE): {mse}')

    # Подсчет остатков
    y_actual = y_test  # Фактические значения целевой переменной на тестовом наборе
    y_predicted = y_pred  # Прогнозные значения целевой переменной на тестовом наборе
    residuals = y_actual - y_predicted  # Остатки

    # Вычисление остаточной дисперсии
    residual_variance = np.var(residuals)
    print(f'Остаточная дисперсия: {residual_variance}')

    # Модель Ridge с кросс-валидацией для выбора параметра alpha (λ)
    alphas = [0.01, 0.1, 1, 10, 100]  # Диапазон значений alpha для перебора

    ridge_model = RidgeCV(alphas=alphas, cv=5)  # cv - количество фолдов в кросс-валидации
    ridge_model.fit(X_train, y_train)  # Обучение модели Ridge

    # Получение наилучшего значения alpha (λ)
    best_alpha_ridge = ridge_model.alpha_

    # Оценка коэффициентов модели Ridge
    ridge_coefficients = ridge_model.coef_

    # Прогнозирование с использованием модели Ridge
    y_pred_ridge = ridge_model.predict(X_test)

    # Вычисление коэффициента детерминации (R^2) и среднеквадратичной ошибки (MSE) для Ridge
    r2_ridge = r2_score(y_test, y_pred_ridge)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)

    # Модель Lasso с кросс-валидацией для выбора параметра alpha (α)
    lasso_model = LassoCV(alphas=alphas, cv=5)  # cv - количество фолдов в кросс-валидации
    lasso_model.fit(X_train, y_train)  # Обучение модели Lasso

    # Получение наилучшего значения alpha (α)
    best_alpha_lasso = lasso_model.alpha_

    # Оценка коэффициентов модели Lasso
    lasso_coefficients = lasso_model.coef_

    # Прогнозирование с использованием модели Lasso
    y_pred_lasso = lasso_model.predict(X_test)

    # Вычисление коэффициента детерминации (R^2) и среднеквадратичной ошибки (MSE) для Lasso
    r2_lasso = r2_score(y_test, y_pred_lasso)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)

    # Вывод результатов
    print(f"Лучшее значение alpha (λ) для Ridge: {best_alpha_ridge}")
    print(f"Коэффициенты модели Ridge: {ridge_coefficients}")
    print(f"R^2 для Ridge: {r2_ridge}")
    print(f"MSE для Ridge: {mse_ridge}")

    print(f"Лучшее значение alpha (α) для Lasso: {best_alpha_lasso}")
    print(f"Коэффициенты модели Lasso: {lasso_coefficients}")
    print(f"R^2 для Lasso: {r2_lasso}")
    print(f"MSE для Lasso: {mse_lasso}")

    # Прогнозирование с использованием модели Ridge
    y_pred_ridge = ridge_model.predict(X_test)

    # Прогнозирование с использованием модели Lasso
    y_pred_lasso = lasso_model.predict(X_test)

    # Вывод результатов прогноза
    print("Прогноз с использованием модели Ridge:")
    print(y_pred_ridge)

    print("Прогноз с использованием модели Lasso:")
    print(y_pred_lasso)
    # # Используйте модель для оценки стоимости автомобиля (пример входных данных)
    # input_data =  [2017, 1435, 1.0, 50000.0, 1.0, 2.0, 120.0, 5.0, 0.0]

    # Расчет MAE для LinearRegression модели
    mae_linear = mean_absolute_error(y_test, y_pred)

    # Расчет MAE для Ridge модели
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

    # Расчет MAE для Lasso модели
    mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

    # Вывод результатов MAE
    print(f"MAE для LinearRegression: {mae_linear}")
    print(f"MAE для Ridge: {mae_ridge}")
    print(f"MAE для Lasso: {mae_lasso}")

    plt.figure()
    # График остатков для LinearRegression модели
    residuals_linear = y_actual - y_pred
    plt.scatter(y_pred, residuals_linear)
    plt.xlabel("Прогнозные значения")
    plt.ylabel("Остатки")
    plt.title("График остатков для LinearRegression модели")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()

    plt.figure()
    # График остатков для Ridge модели
    residuals_ridge = y_actual - y_pred_ridge
    plt.scatter(y_pred_ridge, residuals_ridge)
    plt.xlabel("Прогнозные значения")
    plt.ylabel("Остатки")
    plt.title("График остатков для Ridge модели")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()

    plt.figure()
    # График остатков для Lasso модели
    residuals_lasso = y_actual - y_pred_lasso
    plt.scatter(y_pred_lasso, residuals_lasso)
    plt.xlabel("Прогнозные значения")
    plt.ylabel("Остатки")
    plt.title("График остатков для Lasso модели")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()

    # Для модели Ridge
    residuals_ridge = y_actual - y_pred_ridge  # Остатки для Ridge
    residual_variance_ridge = np.var(residuals_ridge)
    print(f'Остаточная дисперсия для модели Ridge: {residual_variance_ridge}')

    # Для модели Lasso
    residuals_lasso = y_actual - y_pred_lasso  # Остатки для Lasso
    residual_variance_lasso = np.var(residuals_lasso)
    print(f'Остаточная дисперсия для модели Lasso: {residual_variance_lasso}')


    input = [input_data]
    # print(input)
    # Используйте обученную модель для прогноза
    predicted_price = model.predict(input)
    print("predicted_price = model.predict(input)")
    print(predicted_price * 100000 * 1.16)

    # То же самое, что и для LinearRegression, но используйте ridge_model вместо model
    predicted_price_ridge = ridge_model.predict(input)
    print("predicted_price_ridge = ridge_model.predict(input)")
    print(predicted_price_ridge * 100000 * 1.16)

    # То же самое, что и для LinearRegression, но используйте lasso_model вместо model
    predicted_price_lasso = lasso_model.predict(input)
    print("predicted_price_lasso = lasso_model.predict(input)")
    print(predicted_price_lasso * 100000 * 1.16)

    r2_linear = r2_score(y_test, y_pred)
    mse_linear = mean_squared_error(y_test, y_pred)

    r2_ridge = r2_score(y_test, y_pred_ridge)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)

    r2_lasso = r2_score(y_test, y_pred_lasso)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)

    if r2_ridge > r2_linear and r2_ridge > r2_lasso and mse_ridge < mse_linear and mse_ridge < mse_lasso:
        # Вывод результата для Ridge модели
        print("Прогноз с использованием модели Ridge:")
        print(predicted_price_ridge * 100000 * 1.16)
        price_of_the_car = predicted_price_ridge[0] * 100000 * 1.16
        formatted_price = '{:,.2f}'.format(price_of_the_car).replace(',', ' ')
        output_data = f'{formatted_price} ₽'
        return output_data
    elif r2_lasso > r2_linear and r2_lasso > r2_ridge and mse_lasso < mse_linear and mse_lasso < mse_ridge:
        # Вывод результата для Lasso модели
        print("Прогноз с использованием модели Lasso:")
        print(predicted_price_lasso * 100000 * 1.16)
        price_of_the_car = predicted_price_lasso[0] * 100000 * 1.16
        formatted_price = '{:,.2f}'.format(price_of_the_car).replace(',', ' ')
        output_data = f'{formatted_price} ₽'
        return output_data
    else:
        # Вывод результата для LinearRegression модели (если другие модели не превосходят)
        print("Прогноз с использованием модели LinearRegression:")
        print(predicted_price * 100000 * 1.16)
        price_of_the_car = predicted_price[0] * 100000 * 1.16
        formatted_price = '{:,.2f}'.format(price_of_the_car).replace(',', ' ')
        output_data = f'{formatted_price} ₽'
        return output_data

