import pandas as pd

# Define the astronomical start and end dates for seasons.
# Format: (Month, Day)
# Note: Winter spanning December to March requires specific logic in the mapping loop.
SEASON_MAPPING = {
    'winter': ((12, 21), (3, 19)),
    'spring': ((3, 20), (6, 20)),
    'summer': ((6, 21), (9, 22)),
    'autumn': ((9, 23), (12, 20))
}

def _extract_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts core cyclical time components from the 'time' column.
    
    This method decomposes the timestamp into integer components that allow models 
    to identify hourly, weekly, and monthly patterns.
    
    Args:
        data (pd.DataFrame): The input dataframe containing a 'time' column of datetime objects.
        
    Returns:
        pd.DataFrame: Dataframe with added 'hour', 'day_of_week', and 'month' columns.
    """
    data['hour'] = data['time'].dt.hour
    data['day_of_week'] = data['time'].dt.dayofweek # 0=Monday, 6=Sunday
    data['month'] = data['time'].dt.month
    return data

def _standard_work_hours(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a binary flag for standard professional working hours.
    
    Standard work hours are defined here as 09:00 to 17:00 (9 AM to 5 PM).
    This feature helps distinguish between professional productivity and personal time.
    
    Args:
        data (pd.DataFrame): Dataframe with an 'hour' column or 'time' column.
        
    Returns:
        pd.DataFrame: Dataframe with 'is_work_hours' (1 for 9-17, else 0).
    """
    # Vectorized approach is faster than lambda apply
    data['is_work_hours'] = data['time'].dt.hour.between(9, 16).astype(int)
    return data

def _extract_seasonal_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Maps timestamps to their respective meteorological/astronomical seasons.
    
    Seasonal features are critical for behavioral data as they correlate with 
    daylight hours, weather-dependent activity, and mood (e.g., SAD).
    
    Args:
        data (pd.DataFrame): Dataframe containing a 'time' column.
        
    Returns:
        pd.DataFrame: Dataframe with a categorical 'season' column.
    """
    def get_season(date: pd.Timestamp, season_mapping: dict) -> str:
        month_day = (date.month, date.day)
        # Check standard season ranges
        for season, (start, end) in season_mapping.items():
            if start <= month_day <= end:
                return season
        # Fallback for Winter (which wraps around the calendar year 12/21 to 03/19)
        return 'winter'
            
    data['season'] = data['time'].apply(lambda x: get_season(x, SEASON_MAPPING))
    return data

def _add_weekend_feature(data: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies if a record occurred on a weekend.
    
    Weekends (Saturday and Sunday) typically show significantly different 
    behavioral signatures in app usage and social activity compared to weekdays.
    
    Args:
        data (pd.DataFrame): Dataframe containing a 'time' column.
        
    Returns:
        pd.DataFrame: Dataframe with boolean 'is_weekend' column.
    """
    data['is_weekend'] = (data['time'].dt.dayofweek >= 5).astype(int)
    return data

def _add_is_night_time(data: pd.DataFrame) -> pd.DataFrame:
    """
    Flags records occurring during typical sleep or rest hours.
    
    Nighttime is defined as 22:00 (10 PM) to 06:00 (6 AM). This is highly 
    predictive of "Revenge Bedtime Procrastination" or insomnia-related usage.
    
    Args:
        data (pd.DataFrame): Dataframe containing a 'time' column.
        
    Returns:
        pd.DataFrame: Dataframe with 'is_night_time' (1 for night, else 0).
    """
    h = data['time'].dt.hour
    data['is_night_time'] = ((h >= 22) | (h < 6)).astype(int)
    return data

def extract(data: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the full time-based feature engineering pipeline.
    
    This is the high-level entry point that transforms a raw timestamped 
    dataset into a feature-rich representation of temporal context.
    
    Order of operations:
    1. Base time components (Hour, Day, Month)
    2. Seasonal mapping
    3. Weekend/Weekday classification
    4. Nighttime vs. Daytime classification
    5. Professional work hour classification
    
    Args:
        data (pd.DataFrame): The raw input dataframe. Must have a 'time' column.
        
    Returns:
        pd.DataFrame: The enriched dataframe ready for machine learning analysis.
    """
    # Ensure 'time' is in datetime format before processing
    if not pd.api.types.is_datetime64_any_dtype(data['time']):
        data['time'] = pd.to_datetime(data['time'])

    data = _extract_time_features(data)
    data = _extract_seasonal_features(data)
    data = _add_weekend_feature(data)
    data = _add_is_night_time(data)
    data = _standard_work_hours(data)
    
    return data