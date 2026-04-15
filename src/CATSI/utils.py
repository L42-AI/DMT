import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


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


def replace_nan_with_col_mean(x):
    missing_flag = np.isnan(x)
    missing_indices = np.where(missing_flag)
    col_means = np.nanmean(x, axis=0)
    x[missing_indices] = np.take(col_means, missing_indices[1])
    return x, missing_flag


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


def load_raw_data(data_dir, testing):
    def load_ith_pt_training(i):
        """
        i (int): Index for individual. 
        The data is individual data and likely has the shape (Time, 1 + number of 
        Variables). The extra column are the timestamps. 

        Two types of data are extracted, and they are likely two identical datasets, 
        except the second one has some generated missing values, for some evaluation 
        purpose. But according to the name it should be loading training data only. 
        """
        pt_with_na = pd.read_csv(data_dir / 'train_with_missing' / f'{i}.csv').values
        pt_ground_truth = pd.read_csv(data_dir / 'train_groundtruth' / f'{i}.csv').values

        # The timestamp column was saved.
        time_stamps = pt_with_na[:, 0]
        
        # The remaining variables were selected from both datasets
        pt_with_na, pt_ground_truth = pt_with_na[:, 1:], pt_ground_truth[:, 1:]
        pt_with_na, missing_flag = replace_nan_with_col_mean(pt_with_na)
        """ Likely pytorch does not deal well with NaNs for some reason. The author 
        replaces the NaNs with the mean of each variable/column, but keeps a record of 
        the locations in missing_flag. 
        """

        # Note: NaN in ground truth will not be used. Just to avoid NaN in pytorch which does not support nanmean() etc.
        ptmax = np.nanmax(pt_ground_truth, axis=0).reshape(1, -1)
        ptmin = np.nanmin(pt_ground_truth, axis=0).reshape(1, -1)
        """ The author takes the maximum and minimum values of the ground truth dataset, 
        across rows, ignoring NaNs again. The resulting vector is of shape (num_vars,) 
        and gets reshaped to have a row vector shape (1, num_vars), likely to be broadcast 
        later against (T, num_vars). The values are the min/max of all variables. 

        The original data used likely already had NaNs, which are not recoverable, which is
        why the ground truth data ironically includes missing these missing values. 
        """ 

        
        pt_ground_truth, missing_flag_gt = replace_nan_with_col_mean(pt_ground_truth)
        """ The author replaces NaN in ground truth with column means again. missing_flag_gt 
        saves the locations of NaNs. Important because you can't evaluate without a ground 
        truth, so you keep track of missing values.
        """

        eval_mask = (~missing_flag_gt) & missing_flag  # 1: locs masked for eval
        """ Not (Missing in Ground Truth) AND (Missing in Training Data) -> These are of 
        interest in evaluation. 
        """
        # Save all non-missing observations and converts booleans to float
        observed_mask = (~missing_flag).astype(float)  # 1: observed, 0: missing
        eval_mask = eval_mask.astype(float)

        delta = construct_delta_matrix(pt_with_na, time_stamps, observed_mask)
        """ Function call, which requires the training data, the Series of time-stamps and 
        the locations of all NaNs in the training data.
        pt_with_na is not really used, just for shape. Returns matrix of time passed since 
        last observation.
        """

        return {
            'pt_with_na': pt_with_na,
            'pt_ground_truth': pt_ground_truth,
            'time_stamps': time_stamps,
            'observed_mask': observed_mask,
            'eval_mask': eval_mask,
            'pt_max': ptmax,
            'pt_min': ptmin,
            'length': pt_with_na.shape[0],
            'delta': delta,
            'pid': i
        }

    def load_ith_pt_testing(i):
        """ The ground truth data in the training loader was for evaluating the imputation 
        performance of CATSI. So what is this for? Do we have a model?"""
        pt = pd.read_csv(data_dir / f'{i}.csv').values
        time_stamps = pt[:, 0]
        pt = pt[:, 1:]
        ptmax = np.nanmax(pt, axis=0).reshape(1, -1)
        ptmin = np.nanmin(pt, axis=0).reshape(1, -1)
        pt, missing_flag = replace_nan_with_col_mean(pt)
        observed_mask = (~missing_flag).astype(float)
        delta = construct_delta_matrix(pt, time_stamps, observed_mask)
        return {
            'pt_with_na': pt,
            'time_stamps': time_stamps,
            'observed_mask': observed_mask,
            'pt_max': ptmax,
            'pt_min': ptmin,
            'length': pt.shape[0],
            'delta': delta,
            'pid': i

        }

    if not testing:
        num_pts = len(list((data_dir / 'train_with_missing').glob('*.csv')))
        all_pts = [load_ith_pt_training(i) for i in range(1, num_pts+1)]
        return all_pts
        """ Computes the number of individuals via the number of csv files in the folder. Then 
        uses it to return a list of all training dictionaries with data, masks, and statistics. 
        One for each individual.
        """
    else:
        num_pts = len(list(data_dir.glob('*.csv')))
        all_pts_testing = [load_ith_pt_testing(i) for i in range(1, num_pts+1)]
        return all_pts_testing
        """ Same for the testing set (testing here does not refer to the evaluation of the
        imputation, but some other model afterwards).
        """


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
        data = load_raw_data(data_path.parent, testing)
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
        train_set, valid_set = TimeSeriesDataSet(data[:train_size]), TimeSeriesDataSet(data[train_size:])
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
        """ batch_data here is the list of patient dictionaries for the current batch
        """

        lengths = [x['length'] for x in batch_data]
        pids = [x['pid'] for x in batch_data]
        """ The time-length and patient ids for each patient is retrieved
        """

        lengths, data_idx = torch.sort(torch.LongTensor(lengths),
                                       descending=True)
        """ Converts the lengths to integers, then sorts them in descending length. The indexes of the 
        ordered lengths are saved
        """

        batch_data = [batch_data[idx] for idx in data_idx]
        
        pids = [pids[idx] for idx in data_idx]
        """ Re-order both the dictionary and the patient ids according to time-length in descending order"""

        data_dict = {}
        data_dict['values'] = pad_sequence([torch.FloatTensor(x['pt_with_na']) for x in batch_data],
                                           batch_first=True).to(device)
        data_dict['masks'] = pad_sequence([torch.FloatTensor(x['observed_mask']) for x in batch_data],
                                          batch_first=True).to(device)
        data_dict['deltas'] = pad_sequence([torch.FloatTensor(x['delta']) for x in batch_data],
                                           batch_first=True).to(device)
        """ Creates an empty dictionary to store the batched tensors. Author stacks all matrices along the 
        batch dimension (patient), creating tensors of shape (pts, T_max, var). Because batch_data was 
        sorted, the length of time increases along the batch dimension.
        """

        data_dict['time_stamps'] = pad_sequence([torch.FloatTensor(x['time_stamps']) for x in batch_data],
                                                batch_first=True).to(device)
        """ Same for time stamps, but now stacked tensor is (pts, T_max)
        """
        

        data_dict['lengths'] = lengths.to(device)
        data_dict['pids'] = pids
        data_dict['max_vals'] = torch.FloatTensor(np.concatenate([x['pt_max']
                                                                  for x in batch_data])).to(device).unsqueeze(1)
        data_dict['min_vals'] = torch.FloatTensor(np.concatenate([x['pt_min']
                                                                  for x in batch_data])).to(device).unsqueeze(1)
        """ Each patients pt_max and pt_min are of shape(1, vars). Concatenate stacks them into (pts, vars), and 
        unsqueeze(1) makes it (pts, 1, vars), to broadcast against (pts, T_max, vars)
        """

        if not testing:
            data_dict['evals'] = pad_sequence([torch.FloatTensor(x['pt_ground_truth']) for x in batch_data],
                                              batch_first=True).to(device)
            data_dict['eval_masks'] = pad_sequence([torch.FloatTensor(x['eval_mask']) for x in batch_data],
                                                   batch_first=True).to(device)
            """ Add ground truth values and evaluation masks to batch dictionary, if training/validating
            """

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
