import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from meteostat import Point, Daily, Stations
import joblib
import os

def getRecentWeather(lat, lon, days=7):
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)
    
    print(f" Загрузка последних {days} дней данных...")
    
    point = Point(lat, lon)
    
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime(start_date.year, start_date.month, start_date.day)
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime(end_date.year, end_date.month, end_date.day)
    
    try:
        data = Daily(point, start_date, end_date)
        df = data.fetch()
        
        if df is None or df.empty:
            stations = Stations()
            stations = stations.nearby(lat, lon)
            stations = stations.fetch(1)
            
            if stations is not None and not stations.empty:
                station_id = stations.index[0]
                data = Daily(station_id, start_date, end_date)
                df = data.fetch()
        
        if df is None or df.empty:
            return create_test_data(lat, lon, days)
        
        from .dataPreparation import process_dataframe, add_time_features
        df = process_dataframe(df)
        df = add_time_features(df)
        
        return df
        
    except Exception as e:
        print(e)
        return create_test_data(lat, lon, days)

def create_test_data(lat, lon, days):
    print("!!! Используются тестовые данные")
    dates = pd.date_range(
        start=datetime.now().date() - timedelta(days=days),
        periods=days,
        freq='D'
    )
    
    np.random.seed(42)
    base_temp = 20 if '55' in str(lat) else 15
    temps = base_temp + np.random.randn(days) * 5
    
    df = pd.DataFrame({
        'date': dates,
        'tmax': temps + 3,
        'tmin': temps - 3,
        'tavg': temps,
        'wspd': np.random.rand(days) * 5 + 2,
        'prcp': np.random.rand(days) * 2,
        'pres': 1013 + np.random.randn(days) * 10,
        'rhum': 70 + np.random.randn(days) * 15
    })
    
    from .dataPreparation import add_time_features
    df = add_time_features(df)
    
    return df

def predictFuture(model, recent_data, scaler, seq_length=7, pred_days=3):
    model.eval()
    
    # Загружаем информацию о признаках
    feature_info = joblib.load('models/feature_info.pkl')
    all_features = feature_info['all_features']
    
    scaled_data = scaler.transform(recent_data[all_features].values)
    
    if len(scaled_data) < seq_length:
        pad = seq_length - len(scaled_data)
        pad_data = np.tile(scaled_data[0], (pad, 1))
        input_data = np.vstack([pad_data, scaled_data])
    else:
        input_data = scaled_data[-seq_length:]
    
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Обратная нормализация для tavg
    dummy = np.zeros((prediction.shape[1], len(all_features)))
    dummy[:, 2] = prediction.numpy().flatten()  # tavg на позиции 2
    predicted_temps = scaler.inverse_transform(dummy)[:, 2]
     
    last_date = recent_data['date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(pred_days)]
    
    results = pd.DataFrame({
        'date': future_dates,
        'predicted_tavg': np.round(predicted_temps, 1)
    })
    
    return results

def predictWithRecentData(model_path='models/best_model.pth', lat=None, lon=None, 
                          seq_length=7, pred_days=3):
    
    if lat is None or lon is None:
        raise ValueError("нет координат")
    
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    feature_info = joblib.load('models/feature_info.pkl')
    input_size = feature_info['num_features']
    
    from .model import WeatherLSTM
    model = WeatherLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        output_size=pred_days
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Загрузка scaler
    scaler = joblib.load('models/scaler.pkl')
    
    # Получение данных
    recent_data = getRecentWeather(lat, lon, days=seq_length)
    
    # Прогноз
    results = predictFuture(model, recent_data, scaler, seq_length, pred_days)
    
    return results