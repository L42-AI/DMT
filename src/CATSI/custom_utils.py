import pandas as pd
import numpy as np
from pathlib import Path
# from utils import build_data_loader
import torch
import random
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .custom_model import CATSI
from .custom_main import fit

VAR_NAMES = [
    'activity',
    'appCat.builtin',
    'appCat.communication',
    'appCat.entertainment',
    'appCat.finance',
    'appCat.game',
    'appCat.office',
    'appCat.other',
    'appCat.social',
    'appCat.travel',
    'appCat.unknown',
    'appCat.utilities',
    'appCat.weather',
    'screen',
    'circumplex.arousal',
    'circumplex.valence',
    'mood',
    'call',
    'sms',
]

class TimeSeriesDataSet(Dataset):
    """ Subclass of dataset to store the list of dictionaries. __len__ and __getitem__ are custom defined, 
    let's see later how it interacts with Dataloader. __len__ gives number of patients, __getitem__ extracts 
    an individual patient's data. """
    def __init__(self, data):
        super().__init__()
        self.content = data

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        return self.content[idx]

def construct_delta_matrix(time_series, time_stamps, observed_mask):
    """
    Input here is a matrix with shape (T, num_vars), a pd.Series with the T timestamps, and locations
    of all NaNs in the input matrix.  
    """
    # Initialize matrix
    delta = np.zeros(time_series.shape)
    # First time-stamp row will be just zeros.  
    for t in range(1, time_series.shape[0]):
        delta[t, :] = time_stamps[t] - time_stamps[t - 1] + (1 - observed_mask[t, :]) * delta[t - 1, :]
    """ The delta matrix is the time passed since the last observed data (T, num_vars). If the observation
    is missing at t, the time difference at t-1 gets added. The longer data stays missing, the higher 
    time passed since last observed data.
    """
    return delta

def replace_nan_with_col_mean(x):
    missing_flag = np.isnan(x)
    missing_indices = np.where(missing_flag)
    col_means = np.nanmean(x, axis=0)
    x[missing_indices] = np.take(col_means, missing_indices[1])
    return x, missing_flag

def load_raw_data(data_dir):
    def load_ith_pt_training(idx):
        
        pd_data = pd.read_csv(data_dir / f"{idx}.csv")
        loaded_order = pd_data.columns.tolist()[2:]  # skip first two feature columns 'datetime' and 'seconds'
        if loaded_order != VAR_NAMES:
            raise ValueError(f"Column order mismatch for {idx}: {loaded_order}")
        
        orig_data_values = pd_data.values

        print(f"Diagnostics for id {idx}")

        # Date-time values, to be trimmed and not used
        time_stamps = orig_data_values[:, 0]
        print(f"Datetimes:")
        print(time_stamps.shape)

        
        # Distances from timestep 0, to be trimmed and used
        time_distances = orig_data_values[:, 1]
        print(f"Time distances:")
        print(time_distances.shape)


        # Individual data
        values = pd_data.iloc[:, 2:].values
        print(f"Data value shape:")
        print(values.shape)


        # Trimming
        for t in range(values.shape[0]):
            if ~(values[t, :13] == 0).all():
                values = values[t:, :]
                time_distances = time_distances[t:]
                time_stamps = time_stamps[t:]
                break

        for t in range(values.shape[0] - 1, 0, -1):
            if ~np.isnan(values[t, 16]):
                values = values[:t, :]
                time_distances = time_distances[:t]
                time_stamps = time_stamps[:t]
                break

        values, missing_flag = replace_nan_with_col_mean(values)
        
        ptmax = np.nanmax(values, axis=0).reshape(1, -1)
        ptmin = np.nanmin(values, axis=0).reshape(1, -1)

        observed_mask = (~missing_flag).astype(float)
        delta = construct_delta_matrix(values, time_distances, observed_mask)
        print('delta matrix shape')
        print(delta.shape)
        
        return {
            'pt_with_na': values,
            'time_stamps': time_distances,
            'date_time': time_stamps,
            'observed_mask': observed_mask,
            'pt_max': ptmax,
            'pt_min': ptmin,
            'length': values.shape[0],
            'delta': delta,
            'pid': idx
        }

    num_pts = [
        "AS14.01", "AS14.02", "AS14.03", "AS14.05", "AS14.06", "AS14.07", "AS14.08", "AS14.09",
        "AS14.12", "AS14.13", "AS14.14", "AS14.15", "AS14.16", "AS14.17", "AS14.19", "AS14.20",
        "AS14.23", "AS14.24", "AS14.25", "AS14.26", "AS14.27", "AS14.28", "AS14.29", "AS14.30",
        "AS14.31", "AS14.32", "AS14.33"
    ]
    all_pts = [load_ith_pt_training(idx) for idx in num_pts]
    return all_pts

def load_data(data_path, dump_pkl=True, reload_raw=False, testing=False,
              valid_size=0.2, shuffle=True, random_state=None):
    """
    data_path:  The path of the csv, or directory to store pkl
    dump_pkl:   Whether the data should be saved in pkl
    reload_raw: Whether loading the raw data should be enforced. Otherwise it will likely
                load from existing pkl.
    testing:    Whether the training or test set is loaded
    valid_size: The size of the validation set in training, either an integer or a float
                between 0-1.
    """
    
    if not data_path.is_file():
        assert data_path.is_dir(), 'Specified path does not exist.'
        data_path = data_path / f"all_pts_{'training' if not testing else 'testing'}.pkl"
        """ The structure is an indicator of whether to use csv or pkl. If data_path is a file,
        it should then be csv. If it is a directory, the code assumes pkl, and specifies
        the path to the pkl inside the directory."""

    if reload_raw or not data_path.is_file():
        # The data loaded here is either training or test dictionaries, depending on the value of testing.
        data = load_raw_data(data_path.parent)
        if dump_pkl:
            with open(data_path, 'wb') as outfile:
                pickle.dump(data, outfile)
    else:
        with open(data_path, 'rb') as infile:
            data = pickle.load(infile)
    """  Datasets get loaded here, and stored as pkl. There is an option to load from existing pkl
    data here. The data is a bunch of dictionaries (see load_raw_data)
    """

    if not testing:
        if random_state is not None:
            random.seed(random_state)
        if shuffle:
            random.shuffle(data)
        """ Shuffling procedure for training purposes. Remember, data is a list of dictionaries, each dictionary
        represents one patients. So patient order in the list is being shuffled here. The internal ordering within
        each patient is untouched"""

        # If validation size argument is a float between 0-1, it interprets it as a scaling factor. 0.2 would be 20% 
        # of the data
        valid_size = valid_size if valid_size > 1 else int(valid_size * len(data))
        train_size = len(data) - valid_size
        train_set = TimeSeriesDataSet(data[:train_size])
        valid_set = TimeSeriesDataSet(data[train_size:])
        return train_set, valid_set
        """ Assumes valid_size type of either int or a float between 0-1. Anything other will cause index error!! 
        Split of training into training and validation. Then data is stored within a Dataset object, likely for use 
        with Dataloader. The author likely has some custom functionalities within Dataloader, which require this 
        custom structure within TimeSeriesDataSet. """    
    else:
        test_set = TimeSeriesDataSet(data)
        return test_set

def build_data_loader(dataset,
                      device=torch.device('cpu'),
                      batch_size=64,
                      shuffle=True,
                      testing=False):
    def pad_time_series_batch(batch_data):

        lengths = [x['length'] for x in batch_data]
        pids = [x['pid'] for x in batch_data]

        lengths, data_idx = torch.sort(torch.LongTensor(lengths),
                                       descending=True)
        
        batch_data = [batch_data[idx] for idx in data_idx]
        
        pids = [pids[idx] for idx in data_idx]

        data_dict = {}
        data_dict['values'] = pad_sequence([torch.FloatTensor(x['pt_with_na']) for x in batch_data],
                                           batch_first=True).to(device)
        data_dict['masks'] = pad_sequence([torch.FloatTensor(x['observed_mask']) for x in batch_data],
                                          batch_first=True).to(device)
        data_dict['deltas'] = pad_sequence([torch.FloatTensor(x['delta']) for x in batch_data],
                                           batch_first=True).to(device)
        data_dict['time_stamps'] = pad_sequence(
            [torch.from_numpy(np.asarray(x['time_stamps'], dtype=np.float32)) for x in batch_data],
            batch_first=True,
        ).to(device)
        print(f'time_stamps shape: {data_dict['time_stamps'].shape}')

        data_dict['lengths'] = lengths.to(device)
        data_dict['pids'] = pids
        data_dict['max_vals'] = torch.FloatTensor(np.concatenate([x['pt_max']
                                                                  for x in batch_data])).to(device).unsqueeze(1)
        data_dict['min_vals'] = torch.FloatTensor(np.concatenate([x['pt_min']
                                                                  for x in batch_data])).to(device).unsqueeze(1)
        data_dict['date_time'] = [np.asarray(x['date_time']) for x in batch_data]
        
        """ !!! This is likely what gets passed to the forward method of CATSI"""
        return data_dict
        

    data_iter = DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           collate_fn=pad_time_series_batch)
    """ Initialized the dataloader with the custom collate function. With this, dataloader samples batch_size patients
    from the dataset, collects their dictionaries into a list, and runs it through pad_time_series_batch to produce the
    final dictionary to run through CATSI fit"""

    return data_iter


def build_data_loaders(data_path, train_size, batch_size, device, shuffle=True):
    train_set, test_set = load_data(data_path, valid_size=1-train_size)
    train_iter = build_data_loader(train_set, device, batch_size, shuffle)
    test_iter = build_data_loader(test_set, device, batch_size, shuffle)
    return train_iter, test_iter

def _imputed_batch_to_long(batch: dict, imputations: torch.Tensor, only_missing: bool = True) -> pd.DataFrame:
    """
    Convert one batch of model outputs into long-format rows.
 
    args
    ----------
    batch       : the dict produced by pad_time_series_batch
    imputations : (batch_size, T_max, num_vars) tensor — already rescaled
 
    Returns
    -------
    pd.DataFrame with columns [id, time, variable, value]
    """
    rows = []
    for i, pid in enumerate(batch['pids']):
        T = batch['lengths'][i].item()          
        # ts_numeric = batch['date_time'][i, :T].cpu().numpy()   # shape (T,)
        # dates = pd.to_datetime(ts_numeric, unit='s', origin='unix') 
        ts_raw = batch['date_time'][i][:T]
        dates = pd.to_datetime(ts_raw)
        imp = imputations[i, :T, :].cpu().numpy()   # (T, num_vars)
        masks = batch['masks'][i, :T, :].cpu().numpy()
        origs = batch['values'][i, :T, :].cpu().numpy()
 
        for t in range(T):
            for v_idx, var_name in enumerate(VAR_NAMES):
                if only_missing and masks[t, v_idx] == 1:
                    rows.append({
                    'id':       pid,
                    'time':     dates[t],
                    'variable': var_name,
                    'value':    float(origs[t, v_idx]),
                    })
                else:
                    rows.append({
                        'id':       pid,
                        'time':     dates[t],
                        'variable': var_name,
                        'value':    float(imp[t, v_idx]),
                    })
 
    return pd.DataFrame(rows, columns=['id', 'time', 'variable', 'value'])


def catsi_impute(
    data_dir: Path,
    epochs: int = 300,
    batch_size: int = 4,
    valid_size: float = 0.2,
    device: torch.device = torch.device('cpu'),
    reload_raw: bool = False,
) -> pd.DataFrame:
    """
    Train CATSI on the dataset found in *data_dir*, then run imputation on
    all patients and return the result as a long-format DataFrame.
 
    args
    ----------
    data_dir   : directory that contains the per-patient CSV files and will
                 also store the cached .pkl files
    epochs     : training epochs
    batch_size : mini-batch size used for both training and inference
    valid_size : fraction of patients held out for validation during training
    device     : torch device
    reload_raw : force re-reading from raw CSVs (ignores any cached .pkl)
 
    Returns
    -------
    pd.DataFrame with columns ['id', 'time', 'variable', 'value']
        One row per (patient, day, variable).  Values are the CATSI
        imputed estimates (observed values are reproduced as-is by the
        fusion layer: final_imp = mask * observed + (1-mask) * model_imp).
    """
    # ------------------------------------------------------------------
    # 1. Build training / validation iterators
    # ------------------------------------------------------------------
    train_set, valid_set = load_data(
        data_path=data_dir,
        valid_size=valid_size,
        reload_raw=reload_raw,
    )
    train_iter = build_data_loader(train_set, device, batch_size=batch_size, shuffle=True)
    valid_iter = build_data_loader(valid_set, device, batch_size=batch_size, shuffle=False)
 
    num_vars = 19  # 19
    model = CATSI(num_vars=num_vars).to(device)
    fit(model=model, train_iter=train_iter, valid_iter=valid_iter, epochs=epochs, device=device)
 
    total_set, _ = load_data(
        data_path=data_dir,
        valid_size=0,         
        reload_raw=False,      
        shuffle=False,
    )
    total_iter = build_data_loader(
        total_set, device, batch_size=1, shuffle=False
    )
 
    model.eval()
    long_frames = []
    with torch.no_grad():
        for batch in total_iter:
            ret = model(batch)
            # ret['imputations'] shape: (1, T_padded, num_vars) — already rescaled
            frame = _imputed_batch_to_long(batch, ret['imputations'])
            long_frames.append(frame)
 
    imputed_long = (
        pd.concat(long_frames, ignore_index=True)
        .sort_values(['id', 'variable', 'time'])
        .reset_index(drop=True)
    )
    print(imputed_long)
    return imputed_long


if __name__=="__main__":
    
    data_dir = Path("src/data/catsi")
    result = catsi_impute(data_dir, epochs=100, reload_raw=True)
    print(result.head(20))
    print(result.shape)
