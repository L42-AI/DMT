import pandas as pd

def wide_format_daily(data: pd.DataFrame):
    return data.pivot(index = ['id', 'date'], columns = 'variable', values= 'value')
