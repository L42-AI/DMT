import pandas as pd
from .features_behavioural import add_step_behavioural_features
from .features_temporal import add_static_temporal_features, add_rolling_history

def generate_all_features(df: pd.DataFrame, rolling_windows: list[int] = [3, 5]) -> pd.DataFrame:
    """
    Main entry point for generic timeframe feature engineering.
    df must be a wide dataframe sorted chronologically with a MultiIndex ['id', 'time'].
    """
    # 1. Base Step Features (App categories, usage ratios, entropies)
    df = add_step_behavioural_features(df)

    # 2. Static Temporal Features (Calendar context, cyclical patterns)
    df = add_static_temporal_features(df)

    # 3. Define which columns get which rolling treatments
    cols_lag_mean = [
        'work_leisure_ratio',
        'social_interaction_ratio',
        'app_usage_diversity',
        'activity',
    ]

    cols_sum = [
        'screen',
        'call',
        'sms',
        'work_time',
        'leisure_time',
        'social_time',
        'total_app_time'
    ]

    # 4. Apply Temporal Shifts (Lagged context from previous time steps)
    df = add_rolling_history(
        df,
        cols_lag_mean=cols_lag_mean,
        cols_sum=cols_sum,
        windows=rolling_windows
    )

    return df