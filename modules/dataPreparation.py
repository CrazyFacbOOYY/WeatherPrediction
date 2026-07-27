import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from meteostat import Point, Daily, Stations
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

class WeatherDataset(Dataset):
    def __init__(self, data, seq_length=7, pred_days=7):
        self.seq_length = seq_length
        self.pred_days = pred_days
        
        self.X = []
        self.y = []
        
        # Все 6 погодных признаков (индексы 0-5)
        # tmax(0), tmin(1), tavg(2), wspd(3), prcp(4), pres(5)
        self.weather_indices = list(range(6))
        
        for i in range(len(data) - seq_length - pred_days + 1):
            self.X.append(data[i:i+seq_length])
            # Предсказываем все 6 признаков на pred_days дней
            self.y.append(data[i+seq_length:i+seq_length+pred_days, :6])
        
        self.X = torch.FloatTensor(np.array(self.X))
        self.y = torch.FloatTensor(np.array(self.y))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Загрузка данных с meteostat
def loadWeatherData(city_name, lat, lon, start_date, end_date):
    print(f" Загрузка данных для города:{city_name} За период: {start_date} - {end_date}")
    
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime(start_date.year, start_date.month, start_date.day)
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime(end_date.year, end_date.month, end_date.day)
    
    try:
        stations = Stations()
        stations = stations.nearby(lat, lon)
        stations = stations.fetch(5)
        
        if stations is not None and not stations.empty:
            station_id = stations.index[0]
            print(f"   Найдена станция: {station_id}")
            
            data = Daily(station_id, start_date, end_date)
            df = data.fetch()
            
            if df is not None and not df.empty:
                print(f"   Загружено записей: {len(df)}")
                return process_dataframe(df)
    except Exception as e:
        print(f"   X Ошибка при загрузке: {e}")
        
    return create_mock_data(city_name, start_date, end_date)

# на всякий случай
def create_mock_data(city_name, start_date, end_date):
    print("   Генерация данных")
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    days = len(date_range)
    
    if 'Moscow' in city_name or 'москв' in city_name.lower():
        base_temp = 8
        amplitude = 15
        base_pressure = 1013
    else:
        base_temp = 10
        amplitude = 12
        base_pressure = 1015
    
    np.random.seed(42)
    data = []
    
    for i, d in enumerate(date_range):
        day_of_year = d.timetuple().tm_yday
        seasonal = amplitude * np.sin(2 * np.pi * (day_of_year - 30) / 365)
        random_noise = np.random.randn() * 3
        temp = base_temp + seasonal + random_noise
        
        pressure = base_pressure + np.random.randn() * 10 - seasonal * 0.5
        
        data.append({
            'date': d,
            'tmax': temp + np.random.randn() * 1.5 + 3,
            'tmin': temp + np.random.randn() * 1.5 - 3,
            'tavg': temp + np.random.randn() * 0.5,
            'wspd': np.abs(np.random.randn() * 3 + 4),
            'prcp': np.maximum(np.random.randn() * 2 + 1, 0),
            'pres': pressure
        })
    
    df = pd.DataFrame(data)
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.fillna(0)
    return df

def process_dataframe(df):
    if isinstance(df.index, pd.MultiIndex):
        df = df.groupby(level='time').mean()
    
    df.columns = [col.lower() for col in df.columns]
    
    result_df = pd.DataFrame()
    
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'tavg' in col_lower or 'temp' in col_lower or 'mean' in col_lower:
            col_mapping['tavg'] = col
        elif 'tmax' in col_lower or 'max' in col_lower:
            col_mapping['tmax'] = col
        elif 'tmin' in col_lower or 'min' in col_lower:
            col_mapping['tmin'] = col
        elif 'wspd' in col_lower or 'wind' in col_lower or 'speed' in col_lower:
            col_mapping['wspd'] = col
        elif 'prcp' in col_lower or 'precip' in col_lower or 'rain' in col_lower:
            col_mapping['prcp'] = col
        elif 'pres' in col_lower or 'pressure' in col_lower:
            col_mapping['pres'] = col
    
    for target_col, source_col in col_mapping.items():
        result_df[target_col] = df[source_col].values
    
    if 'tavg' not in result_df.columns:
        if 'tmax' in result_df.columns and 'tmin' in result_df.columns:
            result_df['tavg'] = (result_df['tmax'] + result_df['tmin']) / 2
        else:
            result_df['tavg'] = df.iloc[:, 0].values if len(df.columns) > 0 else 0
    
    required_cols = ['tmax', 'tmin', 'tavg', 'wspd', 'prcp', 'pres']
    for col in required_cols:
        if col not in result_df.columns:
            if col == 'tavg' and 'tmax' in result_df.columns and 'tmin' in result_df.columns:
                result_df['tavg'] = (result_df['tmax'] + result_df['tmin']) / 2
            else:
                result_df[col] = 0
    
    result_df = result_df.fillna(method='ffill').fillna(method='bfill')
    result_df = result_df.fillna(0)
    
    if isinstance(df.index, pd.DatetimeIndex):
        result_df['date'] = df.index
    else:
        result_df['date'] = pd.date_range(
            start=datetime.now().date() - timedelta(days=len(result_df)),
            periods=len(result_df),
            freq='D'
        )
    
    result_df = result_df.reset_index(drop=True)
    return result_df

def add_time_features(df):
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def prepareData(city_name, lat, lon, years=10, seq_length=7, pred_days=7, fixed_end_date=None):
    if fixed_end_date is None:
        end_date = datetime(2025, 12, 31)
    else:
        end_date = fixed_end_date
    
    start_date = end_date - timedelta(days=365*years - 1)
    df = loadWeatherData(city_name, lat, lon, start_date, end_date)
    
    if len(df) < seq_length + pred_days:
        raise ValueError(f"Недостаточно данных. Нужно минимум {seq_length + pred_days} дней")
    
    df = add_time_features(df)
    
    dates = df['date'].values
    
    # Признаки: 6 погодных + 6 временных = 12
    weather_features = ['tmax', 'tmin', 'tavg', 'wspd', 'prcp', 'pres']
    time_features = ['sin_day', 'cos_day', 'sin_month', 'cos_month', 'sin_week', 'cos_week']
    all_features = weather_features + time_features
    
    """
    Признаки для обучения:
    Погодные (6): tmax, tmin, tavg, wspd, prcp, pres
    Временные (6): sin_day, cos_day, sin_month, cos_month, sin_week, cos_week
    Всего: 12 признаков
    """

    df_features = df[all_features]
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_features.values)
    
    dataset = WeatherDataset(scaled_data, seq_length, pred_days)
    
    if len(dataset) == 0:
        raise ValueError(f"датасет пуст")
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(dates, 'models/dates.pkl')
    
    feature_info = {
        'all_features': all_features,
        'weather_features': weather_features,
        'time_features': time_features,
        'num_features': len(all_features),
        'num_weather_features': len(weather_features)
    }
    joblib.dump(feature_info, 'models/feature_info.pkl')
    
    """
    Разделение данных:
    Train: 80%
    Val: 10%
    Test: 10%
    """
    
    return train_dataset, val_dataset, test_dataset, scaler

def createDataLoaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader