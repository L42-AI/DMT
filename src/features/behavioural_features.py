import pandas as pd
import numpy as np

def _extract_work_leisure_ratio(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the ratio of "productive" app usage to "leisure" app usage per user.
    
    Behavioral Context:
    This feature serves as a proxy for a user's focus or distraction levels. 
    A higher ratio suggests a productivity-oriented state, while a lower ratio 
    might indicate relaxation, procrastination, or burnout recovery.
    
    Formula: Work Time / (Work Time + Leisure Time)
    
    Args:
        data (pd.DataFrame): The input dataframe containing 'id', 'variable', and 'value' columns.
        
    Returns:
        pd.DataFrame: Dataframe with the new 'work_leisure_ratio' column.
    """
    work_categories = ['appCat.finance', 'appCat.utilities', 'appCat.office']
    leisure_categories = ['appCat.entertainment', 'appCat.game', 'appCat.social']

    # Vectorized approach: Create boolean masks for fast filtering
    is_work = data['variable'].isin(work_categories)
    is_leisure = data['variable'].isin(leisure_categories)

    # Calculate total work and leisure time per ID simultaneously
    work_time_per_id = data[is_work].groupby('id')['value'].sum()
    leisure_time_per_id = data[is_leisure].groupby('id')['value'].sum()

    # Combine into a temporary dataframe to handle the math safely
    totals = pd.DataFrame({
        'work': work_time_per_id,
        'leisure': leisure_time_per_id
    }).fillna(0) # Fill NaNs with 0 in case a user has NO work or NO leisure time
    
    # Calculate ratio, defaulting to 0 if total is 0 to avoid division by zero
    total_time = totals['work'] + totals['leisure']
    ratio_series = np.where(total_time > 0, totals['work'] / total_time, 0)
    
    # Map the calculated ratios back to the main dataframe based on the user ID
    data['work_leisure_ratio'] = data['id'].map(dict(zip(totals.index, ratio_series)))

    return data

def _extract_social_interaction_ratio(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the proportion of a user's total phone time dedicated to social apps.
    
    Behavioral Context:
    This isolates the "Need for Connection" or "Social Density." High social interaction 
    ratios can indicate strong social support networks or, inversely, FOMO (Fear Of Missing Out) 
    and doom-scrolling, depending on the temporal context (e.g., late at night).
    
    Formula: Social App Time / Total App Time
    
    Args:
        data (pd.DataFrame): The input dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with the new 'social_interaction_ratio' column.
    """
    social_categories = ['appCat.social', 'appCat.communication']
    
    is_social = data['variable'].isin(social_categories)
    is_app = data['variable'].str.startswith('appCat.')

    # Aggregate times per ID
    social_time_per_id = data[is_social].groupby('id')['value'].sum()
    total_app_time_per_id = data[is_app].groupby('id')['value'].sum()
    
    # Create a temporary mapping frame
    totals = pd.DataFrame({
        'social': social_time_per_id,
        'total': total_app_time_per_id
    }).fillna(0)

    # Calculate ratio safely
    ratio_series = np.where(totals['total'] > 0, totals['social'] / totals['total'], 0)
    
    # Map back to main dataframe
    data['social_interaction_ratio'] = data['id'].map(dict(zip(totals.index, ratio_series)))

    return data

def _extract_app_usage_diversity(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Shannon Entropy of a user's app category usage.
    
    Behavioral Context:
    Entropy measures predictability and fragmentation. 
    - Low Entropy (~0): The user is highly predictable, spending all their time in 1 or 2 app categories (Deep focus or hyper-fixation).
    - High Entropy: The user's time is evenly distributed across many categories (Fragmented attention, multitasking, or varied tasks).
    
    Formula: $H = -\sum(p_i * \log(p_i))$ where p_i is the proportion of time spent in category i.
    
    Args:
        data (pd.DataFrame): The input dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with the new 'app_usage_diversity' column.
    """
    # Filter for only app categories
    app_data = data[data['variable'].str.startswith('appCat.', na=False)]
    
    if app_data.empty:
        data['app_usage_diversity'] = 0.0
        return data

    # Create a Pivot Table: Rows = Users, Columns = App Categories, Values = Total Time
    user_app_matrix = app_data.pivot_table(
        index='id', 
        columns='variable', 
        values='value', 
        aggfunc='sum', 
        fill_value=0
    )

    # Convert absolute time to probabilities (row-wise normalization)
    row_sums = user_app_matrix.sum(axis=1)
    # Divide each row by its sum. Use np.maximum to prevent division by zero.
    probabilities = user_app_matrix.div(np.maximum(row_sums, 1e-10), axis=0)

    # Calculate Shannon Entropy row-wise
    # Add 1e-10 inside log to prevent log(0) which returns -inf
    entropy_series = -(probabilities * np.log(probabilities + 1e-10)).sum(axis=1)
    
    # Map the calculated entropies back to the main dataframe
    data['app_usage_diversity'] = data['id'].map(entropy_series).fillna(0)

    return data

def extract(data: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the extraction of behavioral clustering features from app usage logs.
    
    This pipeline translates raw app category durations into higher-level psychological 
    proxies (Productivity, Sociability, and Attention Fragmentation).
    
    Args:
        data (pd.DataFrame): The raw input dataframe containing behavioral data.
        
    Returns:
        pd.DataFrame: The enriched dataframe with behavioral ratio features attached.
    """
    data = _extract_work_leisure_ratio(data)
    data = _extract_app_usage_diversity(data)
    data = _extract_social_interaction_ratio(data)
    return data