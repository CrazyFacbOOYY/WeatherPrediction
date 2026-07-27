CITIES = {
    "Москва": {
        "lat": 55.7558,
        "lon": 37.6173,
        "name_en": "Moscow"
    },
    "Екатеринбург": {
        "lat": 56.83,
        "lon": 60.63,
        "name_en": "Yekaterinburg"
    },
    "Уфа": {
        "lat": 54.78,
        "lon": 56.04,
        "name_en": "Ufa"
    },
}

SELECTED_CITY = "Екатеринбург"  # Выбор города

# Параметры обучения
TRAINING_PARAMS = {
    "years": 30,           # Количество лет для обучения
    "epochs": 50,          # Количество эпох
    "hidden_size": 128,    # Размер скрытого слоя
    "num_layers": 3,       # Количество слоев LSTM
    "batch_size": 16,      # Размер батча
    "learning_rate": 0.001 # Скорость обучения
}

def get_city_info(city_name=None):
    """
    city_name: Название города (если None, берется SELECTED_CITY)
    return: Информация о городе с переменными lat, lon, name_en
    """
    if city_name is None:
        city_name = SELECTED_CITY
    
    if city_name not in CITIES:
        raise ValueError("Город не найден")
    
    return CITIES[city_name]

def get_current_city():
    return SELECTED_CITY

def get_available_cities():
    return list(CITIES.keys())