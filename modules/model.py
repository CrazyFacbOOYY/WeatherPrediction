import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeatherLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=3, output_size=7, dropout=0.2):
        """
        input_size: 12 признаков (6 погодных + 6 временных)
        hidden_size: размер скрытого состояния
        num_layers: количество слоев LSTM
        output_size: количество дней прогноза
        """
        super(WeatherLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_weather_features = 6  # tmax, tmin, tavg, wspd, prcp, pres
        
        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Полносвязные слои для прогноза всех признаков. return: 6 признаков * output_size дней
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.num_weather_features * output_size)
        )
        
    def forward(self, x):
        # LSTM, он попроще
        lstm_out, _ = self.lstm(x)
        
        # Берем последний выход
        last_out = lstm_out[:, -1, :]
        
        # Прогноз всех признаков
        output = self.fc(last_out)  # [batch, 6 * 7]
        
        # Преобразуем в [batch, 7 дней, 6 признаков]
        output = output.view(-1, self.output_size, self.num_weather_features)
        
        return output