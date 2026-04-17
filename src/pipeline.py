import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List
from features.features_behavioural import add_step_behavioural_features
from features.feature_pipeline import add_static_temporal_features, add_rolling_history
from models import SimpleMLP, SimpleGRU, RandomClassificationBaseline, RandomRegressionBaseline, XGBoostClassifierWrapper
from xgboost import XGBClassifier

class BasePipeline:
    CLASSIFICATION: bool
    
    def __init__(self, analyser, batch_size=32):
        self.analyser = analyser
        self.batch_size = batch_size
        self.num_users = None
        self.input_dim = None

    def _prepare_base_data(self) -> pd.DataFrame:
        """ Pivots the dataframe, resets ID to a column, and sets Time as the sole index. """
        wide_format = self.analyser.data.pivot_table(index=['time', 'id'], columns='variable', values='value')
        wide_format = wide_format.reset_index(level='id')
        
        wide_format['id'] = wide_format['id'].apply(lambda x: int(str(x)[-2:]))
        wide_format['id'] = wide_format['id'].astype('category').cat.codes
        
        return wide_format.sort_index()

    def _split_x_y_id(self, df: pd.DataFrame):
        """ Splits the enriched dataframe into Features (X), Targets (y), and IDs. """
        id_series = df.pop('id')
        self.num_users = id_series.nunique()
        
        y = df.pop('mood')
        X = df.drop(columns=['circumplex.valence', 'circumplex.arousal'], errors='ignore')
        
        return X, y, id_series

    def _clean_data(self, X_df):
        return X_df

    def _scale_features(self, X_train, X_val, X_test):
        # Implement your scaling logic here
        return X_train, X_val, X_test

    def _engineer_leaky_covariates(self, df: pd.DataFrame) -> pd.DataFrame:
        leaky_columns = ['circumplex.valence', 'circumplex.arousal']
        for col in leaky_columns:
            if col in df.columns:
                df[f'{col}_lag1'] = df.groupby('id')[col].shift(1)
                
        for col in leaky_columns:
            if f'{col}_lag1' in df.columns:
                df[f'{col}_lag1'] = df[f'{col}_lag1'].fillna(0.0)
                
        return df

    # --- ABSTRACT METHODS ---
    def _engineer_features(self, X_df):
        raise NotImplementedError("Subclasses must define how to engineer features.")
    
    def _build_tensors(self, X_df, y_series, id_series):
        raise NotImplementedError()

    # --- OPTIONAL OVERRIDE: Walk-Forward Splits ---
    def get_walk_forward_loaders(self, n_splits=5, gap=2, test_ratio=0.15):
        def make_loader(X_slice, y_slice, id_slice, time_slice, shuffle):
            dataset = TensorDataset(
                torch.tensor(id_slice.copy(), dtype=torch.long),
                torch.tensor(X_slice.copy(), dtype=torch.float32),
                torch.tensor(y_slice.copy(), dtype=y_dtype),
                torch.tensor(time_slice.copy(), dtype=torch.long) # Added Time Tensor
            )
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)
            
        df = self._prepare_base_data()
        df = self._clean_data(df)
        df = self._engineer_features(df)

        X_df, y_series, id_series = self._split_x_y_id(df)

        # Unpack 4 tensors now
        X_arr, y_arr, id_arr, time_arr, y_dtype = self._build_tensors(X_df, y_series, id_series)

        self.input_dim = X_arr.shape[-1]

        # 6. Global Chronological Split (PER USER)
        train_idx, val_idx, test_idx = [], [], []
        train_ratio = 1.0 - test_ratio

        for uid in np.unique(id_arr):
            idx = np.where(id_arr == uid)[0]
            n_user = len(idx)
            train_end = int(n_user * train_ratio)
            val_end = int(n_user * (train_ratio + test_ratio))
            
            train_idx.extend(idx[:train_end])
            val_idx.extend(idx[train_end:val_end])
            test_idx.extend(idx[val_end:])

        X_train, y_train, id_train, time_train = X_arr[train_idx], y_arr[train_idx], id_arr[train_idx], time_arr[train_idx]
        X_val, y_val, id_val, time_val = X_arr[val_idx], y_arr[val_idx], id_arr[val_idx], time_arr[val_idx]
        X_test, y_test, id_test, time_test = X_arr[test_idx], y_arr[test_idx], id_arr[test_idx], time_arr[test_idx]

        X_train, X_val, X_test = self._scale_features(X_train, X_val, X_test)

        tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        folds = []
        for train_index, val_index in tss.split(X_val):
            train_loader = make_loader(X_val[train_index], y_val[train_index], id_val[train_index], time_val[train_index], shuffle=True)
            val_loader = make_loader(X_val[val_index], y_val[val_index], id_val[val_index], time_val[val_index], shuffle=False)
            folds.append((train_loader, val_loader))
            
        test_loader = make_loader(X_test, y_test, id_test, time_test, shuffle=False)

        print(f"\n--- {self.__class__.__name__} Ready ---")
        return folds, test_loader 
    
    def get_dataloaders(self, train_ratio=0.7, val_ratio=0.15):
        df = self._prepare_base_data()
        df = self._clean_data(df)
        df = self._engineer_features(df)

        X_df, y_series, id_series = self._split_x_y_id(df)

        # Unpack 4 tensors now
        X_arr, y_arr, id_arr, time_arr, y_dtype = self._build_tensors(X_df, y_series, id_series)

        self.input_dim = X_arr.shape[-1]

        # 6. Global Chronological Split PER USER
        train_idx, val_idx, test_idx = [], [], []

        for uid in np.unique(id_arr):
            idx = np.where(id_arr == uid)[0]
            n_user = len(idx)
            train_end = int(n_user * train_ratio)
            val_end = int(n_user * (train_ratio + val_ratio))
            
            train_idx.extend(idx[:train_end])
            val_idx.extend(idx[train_end:val_end])
            test_idx.extend(idx[val_end:])

        # Apply indices to all arrays including time
        X_train, y_train, id_train, time_train = X_arr[train_idx], y_arr[train_idx], id_arr[train_idx], time_arr[train_idx]
        X_val, y_val, id_val, time_val = X_arr[val_idx], y_arr[val_idx], id_arr[val_idx], time_arr[val_idx]
        X_test, y_test, id_test, time_test = X_arr[test_idx], y_arr[test_idx], id_arr[test_idx], time_arr[test_idx]

        X_train, X_val, X_test = self._scale_features(X_train, X_val, X_test)

        # 8. Package into PyTorch DataLoaders
        def make_loader(X_slice, y_slice, id_slice, time_slice, shuffle):
            dataset = TensorDataset(
                torch.tensor(id_slice.copy(), dtype=torch.long),
                torch.tensor(X_slice.copy(), dtype=torch.float32),
                torch.tensor(y_slice.copy(), dtype=y_dtype),
                torch.tensor(time_slice.copy(), dtype=torch.long) # Added Time Tensor
            )
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)

        train_loader = make_loader(X_train, y_train, id_train, time_train, shuffle=True)
        val_loader = make_loader(X_val, y_val, id_val, time_val, shuffle=False)
        test_loader = make_loader(X_test, y_test, id_test, time_test, shuffle=False)

        print(f"\n--- {self.__class__.__name__} Ready ---")
        print(f"Batches -> Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
class TabularPipeline(BasePipeline):
    def __init__(self, analyser, batch_size=32, lookahead=1):
        super().__init__(analyser, batch_size)
        self.lookahead = lookahead
        
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_step_behavioural_features(df)
        df = add_static_temporal_features(df)
        df = self._engineer_leaky_covariates(df)
        
        cols_lag_mean = ['work_leisure_ratio', 'activity']
        cols_sum = ['screen', 'call', 'sms']
        df = add_rolling_history(df, cols_lag_mean, cols_sum, windows=[3, 12, 24])
        
        return df
    
    def _build_tensors(self, X_df, y_series, id_series):
        # 1. Shift target backwards by 'lookahead'
        y_future = y_series.groupby(id_series).shift(-self.lookahead)
        
        # Also shift the index (time) backwards so we map the exact future timestamp
        current_times = pd.Series(y_series.index, index=y_series.index)
        time_future = current_times.groupby(id_series).shift(-self.lookahead)
        
        # 2. VALID-TARGET MASK
        valid_mask = ~y_future.isna()
        
        # 3. Filter X, y, IDs, and Times simultaneously
        X_valid = X_df[valid_mask]
        y_valid = y_future[valid_mask]
        id_valid = id_series[valid_mask]
        time_valid = time_future[valid_mask]
        
        # 4. Handle Classification Binning
        if self.CLASSIFICATION:
            y_valid = pd.qcut(y_valid, q=self.num_bins, labels=False, duplicates='drop')
            self.num_classes = len(np.unique(y_valid.dropna()))
            y_dtype = torch.long
        else:
            y_dtype = torch.float32

        # 5. Convert to Numpy arrays
        X_arr = np.nan_to_num(X_valid.values, nan=0.0)
        y_arr = y_valid.values
        id_arr = id_valid.values
        
        # Convert Pandas Timestamps to Unix Seconds
        time_arr = (time_valid.astype('int64') // 10**9).values
        
        if not self.CLASSIFICATION:
            y_arr = y_arr[:, np.newaxis] 
            
        return X_arr, y_arr, id_arr, time_arr, y_dtype


    def build_model(self, hidden_dim: int = 64, embed_dim: int = 5, dropout_rate: float = 0.5):
        return SimpleMLP(
            input_dim=self.input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=self.num_classes, 
            num_users=self.num_users, 
            embed_dim=embed_dim,
            dropout_rate=dropout_rate
        )

    def build_xgboost_model(self, **kwargs):
        """Builds an XGBoost classifier with a wrapper."""
        # Default XGBoost parameters for classification
        defaults = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss'
        }
        # Override defaults with any user-provided kwargs
        defaults.update(kwargs)
        
        return XGBoostClassifierWrapper(**defaults)

class TimeSeriesPipeline(BasePipeline):
    def __init__(self, analyser, seq_len=7, batch_size=32):
        super().__init__(analyser, batch_size)
        self.seq_len = seq_len

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_step_behavioural_features(df)
        df = add_static_temporal_features(df)
        df = self._engineer_leaky_covariates(df)
        return df 
    
    def _build_tensors(self, X_df, y_series, id_series):
        if self.CLASSIFICATION:
            y_series = pd.qcut(y_series, q=self.num_bins, labels=False, duplicates='drop')
            self.num_classes = len(np.unique(y_series.dropna()))
            y_dtype = torch.long
        else:
            y_dtype = torch.float32

        seq_X, seq_y, seq_ids, seq_times = [], [], [], []
        time_index = y_series.index
        
        for user_id in id_series.unique():
            mask = (id_series == user_id)
            user_X = X_df[mask].values
            user_y = y_series[mask].values
            user_times = time_index[mask] 
            
            for i in range(len(user_X) - self.seq_len):
                target_mood = user_y[i + self.seq_len]
                target_time = user_times[i + self.seq_len]
                
                if not np.isnan(target_mood):
                    seq_X.append(user_X[i : i + self.seq_len])
                    seq_y.append(target_mood)
                    seq_ids.append(user_id)
                    seq_times.append(int(target_time.timestamp()))
                    
        X_arr = np.nan_to_num(np.array(seq_X), nan=0.0)
        y_arr = np.array(seq_y)
        id_arr = np.array(seq_ids)
        time_arr = np.array(seq_times)
        
        if not self.CLASSIFICATION:
            y_arr = y_arr[:, np.newaxis] 
            
        return X_arr, y_arr, id_arr, time_arr, y_dtype

    def build_model(self, hidden_dim: int = 64, embed_dim: int = 5, dropout_rate: float = 0.5):
        return SimpleGRU(
            input_dim=self.input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=self.num_classes, 
            num_users=self.num_users, 
            embed_dim=embed_dim,
            dropout_rate=dropout_rate
        )

# --- MIXINS & LEAF CLASSES ---
class ClassificationPipelineMixin:
    CLASSIFICATION = True
    num_users: int
    def __init__(self, *args, num_bins=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bins = num_bins
        self.num_classes = None

    def build_baseline_model(self):
        return RandomClassificationBaseline(
            output_dim=self.num_classes, 
            num_users=self.num_users, 
            embed_dim=5,
        )
    
class RegressionPipelineMixin:
    CLASSIFICATION = False
    num_users: int
    num_classes = 1

    def build_baseline_model(self):
        return RandomRegressionBaseline(
            output_dim=self.num_classes, 
            num_users=self.num_users, 
            embed_dim=5,
        )

class TabularClassification(ClassificationPipelineMixin, TabularPipeline):
    pass

class TimeSeriesClassification(ClassificationPipelineMixin, TimeSeriesPipeline):
    pass

class TabularRegression(RegressionPipelineMixin, TabularPipeline):
    pass

class TimeSeriesRegression(RegressionPipelineMixin, TimeSeriesPipeline):
    pass
