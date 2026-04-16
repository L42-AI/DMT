import pandas as pd
import numpy as np
from .features_math import ensure_columns, safe_divide, rowwise_entropy
from consts import APPCAT_VARS

WORK_COLS = ['appCat.finance', 'appCat.utilities', 'appCat.office']
LEISURE_COLS = ['appCat.entertainment', 'appCat.game', 'appCat.social']
SOCIAL_COLS = ['appCat.social', 'appCat.communication']
OTHER_COLS = ['screen', 'activity', 'call', 'sms']

def add_step_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create behavioural summaries for the current time step.
    Assumes a wide dataframe (index=['id', 'time']).
    """
    df = ensure_columns(df, APPCAT_VARS + OTHER_COLS).copy()
    
    # Fill NAs for feature creation
    df[APPCAT_VARS + OTHER_COLS] = df[APPCAT_VARS + OTHER_COLS].fillna(0)

    # Aggregations for the time step
    df['work_time'] = df[WORK_COLS].sum(axis=1)
    df['leisure_time'] = df[LEISURE_COLS].sum(axis=1)
    df['social_time'] = df[SOCIAL_COLS].sum(axis=1)
    df['total_app_time'] = df[APPCAT_VARS].sum(axis=1)

    # Ratios
    df['work_leisure_ratio'] = safe_divide(
        df['work_time'], 
        df['work_time'] + df['leisure_time']
    )
    df['social_interaction_ratio'] = safe_divide(
        df['social_time'], 
        df['total_app_time']
    )

    # Diversity/Entropy
    df['app_usage_diversity'] = rowwise_entropy(df, APPCAT_VARS)
    
    return df