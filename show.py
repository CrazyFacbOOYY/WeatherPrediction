# показывает доступные записи с meteostat
from datetime import datetime, timedelta
from meteostat import Point, Daily, Stations

def check_available_data():
    # Координаты Города
    lat, lon = 54.78, 56.04
    
    print("\n   Поиск станции...")
    stations = Stations()
    stations = stations.nearby(lat, lon)
    stations = stations.fetch(5)  # Получаем 5 ближайших станций
    
    if stations is None or stations.empty:
        print(" Станции не найдены")
        return
    
    # Показываем найденные станции
    print(f"    Найдено {len(stations)} станций:")
    for idx in stations.index[:3]:
        name = stations.loc[idx, 'name'] if 'name' in stations.columns else 'Unknown'
        print(f"      - {idx}: {name}")
    
    station_id = stations.index[0]
    
    print("\n3. Проверка доступности данных по годам:")
    
    current_year = datetime.now().year
    years_to_check = list(range(1970, current_year + 1))
    
    available_years = []
    
    for year in years_to_check:
        try:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            
            data = Daily(station_id, start_date, end_date)
            df = data.fetch()
            
            if df is not None and not df.empty:
                print(f"   {year}: {len(df)} записей")
                available_years.append(year)
            else:
                print(f"   {year}: Нет данных")
        except Exception as e:
            print(f"   {year}: Ошибка")
    
    # Показываем результат
    if not available_years:
        print("\nДоступных данных не найдено")
        

if __name__ == "__main__":
    check_available_data() # запускается отдельно