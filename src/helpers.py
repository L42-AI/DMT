import pandas as pd
from typing import Dict

def long_to_wide_per_ind(data: pd.DataFrame):
    """
    Returns data in wide format: (id[/time], vars)
    """
    all_vars = data['variable'].unique()
    ind_dict = {}
    for id, group in data.groupby('id'):
        wide = group.pivot(index='time', columns = 'variable', values='value')
        wide = wide.reindex(columns = all_vars)
        ind_dict[id] = wide

    return ind_dict

def wide_to_long_global(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    
    
    long_data = []
    
    for id, wide_data in data_dict.items():
        long = wide_data.reset_index().melt(
            id_vars = 'time',
            var_name = 'variable',
            value_name= 'value'
        )
        long['id'] = id
        long_data.append(long)

    return (pd.concat(long_data, ignore_index=True)
            .loc[:, ['id', 'time', 'variable', 'value']]
            .sort_values(['id', 'variable', 'time']))
def total_time_range(time_series: pd.Series):
    """ Returns the total range of time observed in the whole dataset """
    min_datetime = time_series.dt.date.min()
    max_datetime = time_series.dt.date.max()
    range_datetime = pd.date_range(min_datetime, max_datetime)
    return range_datetime
