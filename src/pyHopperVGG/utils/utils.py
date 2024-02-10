# Description: This file contains the utility functions for reading the HD5 file and the index csv file
import numpy as np
import pandas as pd
from typing import Tuple
import h5py

def read_hd5(data_path: str = None, index_csv_path: str = None) -> Tuple[np.ndarray, pd.DataFrame]:
    # open the hd5 data file and the index csv file
    filename = f'{data_path}'
    _raw_data = np.array(h5py.File(filename, 'r')['dataset0'])
    _index_csv = pd.read_csv(index_csv_path, encoding='utf-8')
    return _raw_data, _index_csv

def get_windows_from_paths(raw_data, index_csv_path):
    raw_data, index_csv = read_hd5(
        data_path=raw_data,
        index_csv_path=index_csv_path,
    )
    windows = []
    for _, row in index_csv.iterrows():
        start_index = row['start']
        end_index = row['end']
        window_data = raw_data[start_index:end_index + 1, :]
        windows.append((window_data, row['class']))
    target = [x[1] for x in windows]
    return windows, target

def load_data_from_hd5(data_path: str, index_csv_path: str):
    """ Reads the HD5, determines the window length and reshapes
        the vector into a NxWxC matrix where N is the number of
        windows, W is the window length and C is the number of
        channels
        """
    _raw_data, _index_csv = read_hd5(data_path, index_csv_path)
    num_chans = _raw_data.shape[1]
    win_len = np.max((_index_csv.end-_index_csv.start+1).values)

    # drop len(window) not equal win_len, and align with labels
    dropped = []
    n = 0
    HData = np.empty((len(_index_csv), win_len, num_chans))
    for k in range(0, len(_index_csv)):
        start = _index_csv.start[k]  # 1-based?
        end = _index_csv.end[k]
        if (end - start + 1) == win_len:
            HData[n] = _raw_data[start-1:end]
            n += 1
        else:
            dropped += [k]

    HData   = HData[0:n]
    HLabels = np.delete(np.array(_index_csv['class']), dropped)
    HGroups = np.delete(np.array(_index_csv['group']), dropped)
    return HData, HLabels, HGroups