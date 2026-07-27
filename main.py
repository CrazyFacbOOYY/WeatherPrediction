import torch
import argparse
from modules.dataPreparation import prepareData, createDataLoaders
from modules.model import WeatherLSTM
from modules.train import trainModel, evaluateModel
import os
import json
from datetime import datetime
from config import get_city_info, get_current_city, TRAINING_PARAMS

def main():

    # outdated // не используются, лучше менять config и заускать как есть.
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default=None, 
                        help='Название города (если не указан, берется из config.py)')
    parser.add_argument('--lat', type=float, default=None, help='Широта')
    parser.add_argument('--lon', type=float, default=None, help='Долгота')
    parser.add_argument('--years', type=int, default=None, help='Количество лет для обучения')
    parser.add_argument('--seq_length', type=int, default=7, help='Дней для анализа (всегда 7)')
    parser.add_argument('--pred_days', type=int, default=7, help='Дней прогноза (всегда 7)')
    parser.add_argument('--epochs', type=int, default=None, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=None, help='Размер батча')
    parser.add_argument('--hidden_size', type=int, default=None, help='Размер скрытого слоя')
    parser.add_argument('--num_layers', type=int, default=None, help='Количество слоев LSTM')
    parser.add_argument('--learning_rate', type=float, default=None, help='Скорость обучения')
    
    args = parser.parse_args()
    
    # Определяем город
    if args.city:
        city_name = args.city
    else:
        city_name = get_current_city()
    

    city_info = get_city_info(city_name)
    lat = args.lat if args.lat is not None else city_info['lat']
    lon = args.lon if args.lon is not None else city_info['lon']
    city_name_en = city_info['name_en']
    
    # берем из config 
    years = args.years if args.years is not None else TRAINING_PARAMS['years']
    epochs = args.epochs if args.epochs is not None else TRAINING_PARAMS['epochs']
    batch_size = args.batch_size if args.batch_size is not None else TRAINING_PARAMS['batch_size']
    hidden_size = args.hidden_size if args.hidden_size is not None else TRAINING_PARAMS['hidden_size']
    num_layers = args.num_layers if args.num_layers is not None else TRAINING_PARAMS['num_layers']
    learning_rate = args.learning_rate if args.learning_rate is not None else TRAINING_PARAMS['learning_rate']
    
    seq_length = 7
    pred_days = 7
    
    os.makedirs('models', exist_ok=True)
    
    params = {
        'city': city_name,
        'city_en': city_name_en,
        'lat': lat,
        'lon': lon,
        'years': years,
        'seq_length': seq_length,
        'pred_days': pred_days,
        'epochs': epochs,
        'batch_size': batch_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'learning_rate': learning_rate
    }
    
    with open('models/model_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    end_year = 2025
    start_year = end_year - years + 1
    
    print(f"  Обучение модели для города: {city_name.upper()}")
    print(f"  Период: {start_year} - {end_year} ({years} лет)")
    print(f"  Координаты: {lat}, {lon}")
    print(f" {hidden_size} нейронов, {num_layers} слоев, Эпох: {epochs}")
    
    try:
        end_date = datetime(end_year, 12, 31)
        start_date = datetime(start_year, 1, 1)
        
        train_dataset, val_dataset, test_dataset, scaler = prepareData(
            city_name, lat, lon, years,
            seq_length, pred_days,
            fixed_end_date=end_date
        )
    except Exception as e:
        print(e)
        return
    
    train_loader, val_loader, test_loader = createDataLoaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )
    
    device = torch.device('cpu')
    model = WeatherLSTM(
        input_size=12,      # 6 погодных + 6 временных
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=pred_days
    )
    
    # Обучение
    trainModel(
        model, train_loader, val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device
    )
    
    # Оценка
    evaluateModel(model, test_loader, scaler, device)

if __name__ == "__main__":
    main()