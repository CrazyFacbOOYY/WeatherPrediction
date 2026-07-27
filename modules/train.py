import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import joblib

def trainModel(model, train_loader, val_loader, epochs=100, learning_rate=0.001, device='cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    
    device_str = str(device).upper()
    print(f" Using: {device_str}")
    print(f"Параметров модели: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)  # [batch, 7 дней, 6 признаков]
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
            }, 'models/best_model.pth')
            print(f' Чекпоинт. Модель сохранена при val_loss: {avg_val_loss:.6f}')
        
        scheduler.step(avg_val_loss)
    return None, None

def evaluateModel(model, test_loader, scaler, device='cpu'):
    model.eval()
    model.to(device)
    
    feature_info = joblib.load('models/feature_info.pkl')
    num_features = feature_info['num_features']
    num_weather = feature_info['num_weather_features']
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)  # [batch, 7, 6]
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions)  # [n_samples, 7, 6]
    actuals = np.array(actuals)          # [n_samples, 7, 6]
    
    # Обратная нормализация для каждого признака
    predictions_denorm = []
    actuals_denorm = []
    
    for i in range(num_weather):
        # Создаем фиктивный массив для обратного преобразования
        dummy_pred = np.zeros((predictions.shape[0], predictions.shape[1], num_features))
        dummy_pred[:, :, i] = predictions[:, :, i]
        dummy_actual = np.zeros((actuals.shape[0], actuals.shape[1], num_features))
        dummy_actual[:, :, i] = actuals[:, :, i]
        
        # Обратная нормализация для каждого дня
        pred_flat = dummy_pred.reshape(-1, num_features)
        actual_flat = dummy_actual.reshape(-1, num_features)
        
        try:
            pred_denorm = scaler.inverse_transform(pred_flat)[:, i].reshape(predictions.shape[0], predictions.shape[1])
            actual_denorm = scaler.inverse_transform(actual_flat)[:, i].reshape(actuals.shape[0], actuals.shape[1])
        except:
            pred_denorm = predictions[:, :, i]
            actual_denorm = actuals[:, :, i]
        
        predictions_denorm.append(pred_denorm)
        actuals_denorm.append(actual_denorm)
    
    predictions_denorm = np.array(predictions_denorm)  # [6, n_samples, 7]
    actuals_denorm = np.array(actuals_denorm)          # [6, n_samples, 7]
    
    # Вычисляем метрики для каждого признака
    feature_names = ['tmax', 'tmin', 'tavg', 'wspd', 'prcp', 'pres']
    
    total_mae = 0
    for i, name in enumerate(feature_names):
        mae = np.mean(np.abs(predictions_denorm[i] - actuals_denorm[i]))
        rmse = np.sqrt(np.mean((predictions_denorm[i] - actuals_denorm[i])**2))
        print(f"   {name}: MAE={mae:.2f}, RMSE={rmse:.2f}")
        total_mae += mae
    
    avg_mae = total_mae / len(feature_names)
    
    return predictions_denorm, actuals_denorm, avg_mae