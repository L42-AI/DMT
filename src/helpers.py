import pandas as pd

def wide_format_daily(data: pd.DataFrame):
    return data.pivot(index = ['id', 'date'], columns = 'variable', values= 'value')

def total_time_range(time_series: pd.Series):
    """ Returns the total range of time observed in the whole dataset """
    min_datetime = time_series.dt.date.min()
    max_datetime = time_series.dt.date.max()
    range_datetime = pd.date_range(min_datetime, max_datetime)
    return range_datetime