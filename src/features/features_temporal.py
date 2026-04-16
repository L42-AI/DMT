import pandas as pd
import numpy as np

# --- 1. STATIC TEMPORAL FEATURES ---
SEASON_MAPPING = {
    'winter': ((12, 21), (3, 19)),
    'spring': ((3, 20), (6, 20)),
    'summer': ((6, 21), (9, 22)),
    'autumn': ((9, 23), (12, 20))
}

def _get_season(date: pd.Timestamp) -> str:
    month_day = (date.month, date.day)
    for season, (start, end) in SEASON_MAPPING.items():
        if start <= month_day <= end:
            return season
    return 'winter'

def add_static_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Extracts calendar context from the Time index and groups by the 'id' column. """
    df = df.copy()
    time_s = df.index # This is now exclusively your time array
    
    # 1. Basic Calendar Features
    df['day_of_week'] = time_s.dayofweek
    df['day_of_month'] = time_s.day
    df['month'] = time_s.month
    df['week_of_year'] = time_s.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 2. Cyclical Encodings
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # 3. Time Since Start (Using the 'id' column to group)
    temp_time_series = pd.Series(time_s, index=df.index)
    first_dates = temp_time_series.groupby(df['id']).transform('min')
    df['days_since_start'] = (time_s - first_dates).dt.total_seconds() / 86400.0

    # 4. Seasonality (One-Hot Encoded)
    season_series = pd.Series(time_s.map(_get_season), index=df.index)
    season_dummies = pd.get_dummies(season_series, prefix='season', dtype=float)
    df = pd.concat([df, season_dummies], axis=1)

    return df

# --- 2. ROLLING HISTORICAL FEATURES ---
def add_rolling_history(
    df: pd.DataFrame,
    cols_lag_mean: list[str],
    cols_sum: list[str],
    windows: list[int] = [3, 5]
) -> pd.DataFrame:
    """ Create lagged/rolling features grouped by the 'id' column. """
    df = df.copy()
    
    # Crucial: Sort by ID first, then Time, so rolling windows don't bleed across users
    # In newer Pandas, you can sort by an index name ('time') and a column ('id') simultaneously
    df = df.sort_values(['id', 'time'])

    # Handle Lag/Mean/Std variables
    for col in cols_lag_mean:
        grouped = df.groupby('id')[col]
        shifted = grouped.shift(1)
        df[f'{col}_lag1'] = shifted

        for w in windows:
            rolling_view = shifted.rolling(window=w, min_periods=w)
            df[f'{col}_mean_w{w}'] = rolling_view.mean().reset_index(level=0, drop=True)
            df[f'{col}_std_w{w}'] = rolling_view.std().reset_index(level=0, drop=True)

    # Handle Sum/Mean variables
    for col in cols_sum:
        grouped = df.groupby('id')[col]
        shifted = grouped.shift(1)
        df[f'{col}_lag1'] = shifted

        for w in windows:
            rolling_view = shifted.rolling(window=w, min_periods=w)
            df[f'{col}_sum_w{w}'] = rolling_view.sum().reset_index(level=0, drop=True)
            df[f'{col}_mean_w{w}'] = rolling_view.mean().reset_index(level=0, drop=True)

    return df