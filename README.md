# WeatherPrediction - предсказание погоды на последущие по архивным погодным признакам.

## Установка зависимостей (в терминале):
### Из requirements:
>pip install -r requirements.txt

### Более надежный вариант (у меня были какие-то проблемы с dll файлами, думаю это связано с тем что я использую cpu,а не cuda):
- устанавливаем Torch:
>pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

- остальное:
>pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0 meteostat==1.6.5 streamlit==1.28.1 plotly==5.17.0 tqdm==4.65.0 joblib==1.3.2

## Запуск:
1. Настройте config.py (широта, долгота, эпохи, срок)

2. Обчите модель:
>python main.py

2. Запустите приложение:
>streamlit run app.py
