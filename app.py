import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import json
import warnings
import os
warnings.filterwarnings('ignore')

from config import get_city_info, get_current_city, get_available_cities, CITIES

st.set_page_config(page_title="Прогноз температуры", page_icon="🌡️", layout="wide")
st.title("🌡️ Прогноз погоды на 7 дней")
st.markdown("---")

# Получение данных за последние дни через Open-Meteo
def get_weather_from_openmeteo(lat, lon, days=7):
    try:
        import openmeteo_requests
        
        client = openmeteo_requests.Client()
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days-1)
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": [
                "temperature_2m_max", 
                "temperature_2m_min", 
                "temperature_2m_mean", 
                "wind_speed_10m_max", 
                "precipitation_sum"
            ],
            "timezone": "Europe/Moscow"
        }
        
        responses = client.weather_api(url, params=params)
        response = responses[0]
        daily = response.Daily()
        
        dates = pd.date_range(
            start=start_date, 
            periods=len(daily.Variables(0).ValuesAsNumpy()), 
            freq='D'
        )
        
        df = pd.DataFrame({
            'date': dates,
            'tmax': daily.Variables(0).ValuesAsNumpy(),
            'tmin': daily.Variables(1).ValuesAsNumpy(),
            'tavg': daily.Variables(2).ValuesAsNumpy(),
            'wspd': daily.Variables(3).ValuesAsNumpy(),
            'prcp': daily.Variables(4).ValuesAsNumpy(),
        })
        
        # Давление генерируется, лучше заполнять его собственноручно (Open-Meteo не дает его напрямую)
        df['pres'] = 1013 - (df['tavg'] - 10) * 0.3 + np.random.randn(len(df)) * 3
        df['pres'] = df['pres'].clip(950, 1050)
        
        return df, None
        
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_model():
    try:
        from modules.model import WeatherLSTM
        
        if not os.path.exists('models/best_model.pth'):
            st.error("Модель не найдена. Попробуйте для начала запустить файл main.py")
            return None, None, None, None
        
        feature_info = joblib.load('models/feature_info.pkl')
        input_size = feature_info['num_features']
        
        with open('models/model_params.json', 'r') as f:
            params = json.load(f)

        # Москва, lat lon на всякий случай
        city_name = params.get('city', 'Москва')
        lat = params.get('lat', 55.7558)
        lon = params.get('lon', 37.6173)
        
        model = WeatherLSTM(
            input_size=input_size,
            hidden_size=params.get('hidden_size', 128),
            num_layers=params.get('num_layers', 3),
            output_size=7
        )
        
        checkpoint = torch.load('models/best_model.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler = joblib.load('models/scaler.pkl')
        
        st.success(f"Модель загружена для города {city_name}")
        return model, scaler, feature_info, (city_name, lat, lon)
        
    except Exception as e:
        st.error(f"Ошибка при загрузки модели: {e}")
        return None, None, None, None

# Загрузка модели
if 'model_loaded' not in st.session_state:
    with st.spinner("Загрузка модели..."):
        model, scaler, feature_info, city_info = load_model()
        if model is not None:
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.feature_info = feature_info
            st.session_state.city_info = city_info
            st.session_state.model_loaded = True

# Основная логика
if st.session_state.get('model_loaded', False):
    # Показываем информацию о городе
    city_name, lat, lon = st.session_state.city_info
    
    st.info(f"Текущий город: **{city_name}** (координаты: {lat}, {lon})")
    st.subheader("📝 Введите данные за последние 7 дней")
    
    today = datetime.now().date()
    dates = [today - timedelta(days=i) for i in range(7, 0, -1)]
    
    features = ['tmax', 'tmin', 'tavg', 'wspd', 'prcp', 'pres']
    feature_labels = {
        'tmax': 'Макс. температура (°C)',
        'tmin': 'Мин. температура (°C)',
        'tavg': 'Средняя температура (°C)',
        'wspd': 'Скорость ветра (м/с)',
        'prcp': 'Осадки (мм)',
        'pres': 'Давление (гПа)'
    }
    
    input_data = []
    for i, date in enumerate(dates):
        row = {'Дата': date.strftime('%d.%m.%Y')}
        for feat in features:
            row[feat] = None
        input_data.append(row)
    
    df_input = pd.DataFrame(input_data)
    
    # Кнопки управления
    col_buttons1, col_buttons2, col_buttons3 = st.columns([2, 1, 1])
    
    with col_buttons1:
        if st.button("🌤️ Заполнить автоматически с Open-Meteo", type="primary", use_container_width=True):
            with st.spinner("⏳ Загрузка данных..."):
                df_weather, error = get_weather_from_openmeteo(lat, lon, days=7)
                
                if df_weather is not None:
                    for i, row in df_weather.iterrows():
                        for feat in features:
                            if feat in df_weather.columns:
                                df_input.loc[i, feat] = float(row[feat])
                    
                    st.session_state.df_input = df_input
                    st.success(f"✅ Данные загружены для {city_name}!")
                    st.rerun()
                else:
                    st.error(f"Ошибка загрузки: {error}")
    
    with col_buttons2:
        if st.button("🗑️ Очистить", use_container_width=True):
            for i in range(len(df_input)):
                for feat in features:
                    df_input.loc[i, feat] = None
            st.session_state.df_input = df_input
            st.rerun()
    
    with col_buttons3:
        if st.button("📊 Пример", use_container_width=True):
            np.random.seed(42)
            for i in range(len(df_input)):
                base_temp = 20 + np.random.randn() * 5
                df_input.loc[i, 'tmax'] = round(base_temp + np.random.randn() * 2 + 3, 1)
                df_input.loc[i, 'tmin'] = round(base_temp + np.random.randn() * 2 - 3, 1)
                df_input.loc[i, 'tavg'] = round(base_temp + np.random.randn() * 1, 1)
                df_input.loc[i, 'wspd'] = round(abs(np.random.randn() * 3 + 4), 1)
                df_input.loc[i, 'prcp'] = round(max(0, np.random.randn() * 2 + 1), 1)
                df_input.loc[i, 'pres'] = round(1013 + np.random.randn() * 10, 1)
            st.session_state.df_input = df_input
            st.rerun()
    
    if 'df_input' in st.session_state:
        df_input = st.session_state.df_input
    
    edited_df = st.data_editor(
        df_input,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Дата": st.column_config.TextColumn("Дата", disabled=True),
            "tmax": st.column_config.NumberColumn("Макс. температура (°C)", min_value=-40, max_value=45, step=0.5),
            "tmin": st.column_config.NumberColumn("Мин. температура (°C)", min_value=-40, max_value=45, step=0.5),
            "tavg": st.column_config.NumberColumn("Средняя температура (°C)", min_value=-40, max_value=45, step=0.5),
            "wspd": st.column_config.NumberColumn("Скорость ветра (м/с)", min_value=0, max_value=30, step=0.5),
            "prcp": st.column_config.NumberColumn("Осадки (мм)", min_value=0, max_value=100, step=0.1),
            "pres": st.column_config.NumberColumn("Давление (гПа)", min_value=950, max_value=1050, step=0.5),
        },
        hide_index=True
    )
    
    st.session_state.df_input = edited_df
    
    filled_count = edited_df[features].notna().sum().sum()
    total_cells = len(edited_df) * len(features)
    fill_percentage = (filled_count / total_cells) * 100
    
    col_info1, col_info2, col_info3 = st.columns([2, 1, 1])
    with col_info1:
        st.progress(fill_percentage / 100, text=f"Заполнено {int(fill_percentage)}% данных ({filled_count}/{total_cells} ячеек)")
    with col_info2:
        if fill_percentage < 30:
            st.warning("⚠️ Заполните больше данных для точного прогноз")
        elif fill_percentage < 70:
            st.info("ℹ️ Хорошо, можно делать прогноз")
        else:
            st.success("✅ Все ячейки заполнены")
    
    if st.button("🔮 Сделать прогноз", type="primary", use_container_width=True):
        if filled_count == 0:
            st.error("❌ Заполните хотя бы несколько полей для прогноза")
            st.stop()
        
        with st.spinner("⏳ Прогнозируем..."):
            try:
                for feat in features:
                    if edited_df[feat].isna().all():
                        edited_df[feat] = 0
                    else:
                        edited_df[feat] = edited_df[feat].interpolate(method='linear', limit_direction='both')
                        edited_df[feat] = edited_df[feat].fillna(edited_df[feat].mean())
                
                # Добавляем временные признаки
                all_features = st.session_state.feature_info['all_features']
                
                df_processed = pd.DataFrame()
                df_processed['date'] = pd.to_datetime(dates)
                
                for feat in features:
                    df_processed[feat] = edited_df[feat].values
                
                df_processed['day_of_year'] = df_processed['date'].dt.dayofyear
                df_processed['month'] = df_processed['date'].dt.month
                df_processed['day_of_week'] = df_processed['date'].dt.dayofweek
                
                df_processed['sin_day'] = np.sin(2 * np.pi * df_processed['day_of_year'] / 365.25)
                df_processed['cos_day'] = np.cos(2 * np.pi * df_processed['day_of_year'] / 365.25)
                df_processed['sin_month'] = np.sin(2 * np.pi * df_processed['month'] / 12)
                df_processed['cos_month'] = np.cos(2 * np.pi * df_processed['month'] / 12)
                df_processed['sin_week'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
                df_processed['cos_week'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)
                
                scaled_data = st.session_state.scaler.transform(df_processed[all_features].values)
                
                # Прогноз всех признаков
                input_tensor = torch.FloatTensor(scaled_data).unsqueeze(0)
                with torch.no_grad():
                    prediction = st.session_state.model(input_tensor)
                
                # Обратная нормализация для всех признаков
                feature_info = st.session_state.feature_info
                num_features = feature_info['num_features']
                num_weather = feature_info['num_weather_features']
                
                prediction_np = prediction.numpy()[0]
                
                dummy = np.zeros((7, num_features))
                for i in range(num_weather):
                    dummy[:, i] = prediction_np[:, i]
                
                forecast_full = st.session_state.scaler.inverse_transform(dummy)
                
                # Извлекаем прогнозы
                forecast_data = []
                future_dates = [today + timedelta(days=i+1) for i in range(7)]
                
                for i in range(7):
                    row = {
                        'Дата': future_dates[i].strftime('%d.%m.%Y'),
                        'tmax': round(forecast_full[i, 0], 1),
                        'tmin': round(forecast_full[i, 1], 1),
                        'tavg': round(forecast_full[i, 2], 1),
                        'wspd': round(max(0, forecast_full[i, 3]), 1),
                        'prcp': round(max(0, forecast_full[i, 4]), 1),
                        'pres': round(forecast_full[i, 5], 1)
                    }
                    forecast_data.append(row)
                
                df_forecast = pd.DataFrame(forecast_data)
                
                st.success(f"✅ Прогноз для {city_name} выполнен!")
                
                # Метрики
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Текущая средняя", f"{edited_df['tavg'].iloc[-1]:.1f}°C")
                with col2:
                    st.metric("Средняя прогноз", f"{df_forecast['tavg'].mean():.1f}°C")
                with col3:
                    st.metric("Максимальная", f"{df_forecast['tavg'].max():.1f}°C")
                with col4:
                    st.metric("Минимальная", f"{df_forecast['tavg'].min():.1f}°C")
                
                # Таблица прогноза
                st.subheader("📊 Детальный прогноз на 7 дней")
                
                st.dataframe(
                    df_forecast.style.format({
                        'tavg': '{:.1f}',
                        'tmax': '{:.1f}',
                        'tmin': '{:.1f}',
                        'wspd': '{:.1f}',
                        'prcp': '{:.1f}',
                        'pres': '{:.1f}'
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Дата": st.column_config.TextColumn("Дата"),
                        "tavg": st.column_config.NumberColumn("Средняя (°C)", format="%.1f"),
                        "tmax": st.column_config.NumberColumn("Макс. (°C)", format="%.1f"),
                        "tmin": st.column_config.NumberColumn("Мин. (°C)", format="%.1f"),
                        "wspd": st.column_config.NumberColumn("Ветер (м/с)", format="%.1f"),
                        "prcp": st.column_config.NumberColumn("Осадки (мм)", format="%.1f"),
                        "pres": st.column_config.NumberColumn("Давление (гПа)", format="%.1f"),
                    }
                )
                
                # График температур
                st.subheader("📈 Визуализация прогноза температуры")
                fig = go.Figure()
                
                hist_dates = pd.to_datetime(dates)
                fig.add_trace(go.Scatter(
                    x=hist_dates,
                    y=edited_df['tavg'],
                    mode='lines+markers',
                    name='Фактические данные',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ))
                
                future_dates_dt = pd.to_datetime(future_dates)
                fig.add_trace(go.Scatter(
                    x=future_dates_dt,
                    y=df_forecast['tavg'],
                    mode='lines+markers',
                    name='Прогноз',
                    line=dict(color='red', width=3, dash='dash'),
                    marker=dict(size=8, color='red')
                ))
                
                fig.add_shape(
                    type="line",
                    x0=today,
                    y0=min(edited_df['tavg'].min(), df_forecast['tavg'].min()) - 5,
                    x1=today,
                    y1=max(edited_df['tavg'].max(), df_forecast['tavg'].max()) + 5,
                    line=dict(color="gray", width=2, dash="dash")
                )
                
                fig.add_annotation(
                    x=today,
                    y=max(edited_df['tavg'].max(), df_forecast['tavg'].max()) + 3,
                    text="Сейчас",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-30
                )
                
                fig.update_layout(
                    height=450,
                    title=f"Прогноз температуры для {city_name} на 7 дней",
                    xaxis_title="Дата",
                    yaxis_title="Температура (°C)",
                    hovermode='x unified',
                    showlegend=True,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # График всех признаков
                st.subheader("📈 Визуализация всех погодных параметров")
                
                fig2 = go.Figure()
                
                for feat in ['tavg', 'tmax', 'tmin', 'wspd', 'prcp']: # удрал pres (давление)
                    fig2.add_trace(go.Scatter(
                        x=future_dates_dt,
                        y=df_forecast[feat],
                        mode='lines+markers',
                        name=feat
                    ))
                
                fig2.update_layout(
                    height=400,
                    title=f"Прогноз всех параметров для {city_name}",
                    xaxis_title="Дата",
                    yaxis_title="Значение",
                    hovermode='x unified',
                    showlegend=True,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Скачивание CSV
                csv = df_forecast.to_csv(index=False)
                st.download_button(
                    label="📥 Скачать прогноз (CSV)",
                    data=csv,
                    file_name=f"forecast_{city_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Ошибка: {e}")
                import traceback
                st.code(traceback.format_exc())

else:
    st.warning("""
    ### Модель не загружена :(
    
    Укажите город в файле config.py (строка SELECTED_CITY)
    
    Затем в терминале:
    >python main.py""")