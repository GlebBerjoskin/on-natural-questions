import multiprocessing
import itertools
import torch

import numpy as np

from typing import List, Dict, Tuple, Callable
from pandas.io.json._json import JsonReader
from torch.utils.data import Dataset


def get_lines_quantity(file_path: str) -> int:
    """
    Returns num of lines in a file

    :param file_path: string containing path to the file

    :return: num_lines: number of lines in the file
    """

    num_lines = 0

    with open(file_path) as f:
        for line in f:
            num_lines += 1
    return num_lines


class NQJsonReader(JsonReader):
    """
    Modifies pandas JsonReader class, adding chunk preprocessing, done with multiprocessing pool.
    This allows to prepare data for training/inference on less powerful machines.
    """

    def __init__(self, filepath_or_buffer: str, prepare_nq_samples: Callable, orient: str = None, typ: str = 'frame',
                 dtype: bool = None, convert_axes: bool = None, convert_dates: bool = True,
                 keep_default_dates: bool = True, numpy: bool = False, precise_float: bool = False,
                 date_unit: str = None, encoding: str = None, lines: bool = True, chunksize: int = 2000,
                 compression: str = None, nrows: int = None):
        super(NQJsonReader, self).__init__(str(filepath_or_buffer), orient=orient, typ=typ, dtype=dtype,
                                           convert_axes=convert_axes, convert_dates=convert_dates,
                                           keep_default_dates=keep_default_dates, numpy=numpy,
                                           precise_float=precise_float, date_unit=date_unit, encoding=encoding,
                                           lines=lines, chunksize=chunksize, compression=compression, nrows=nrows)

        self.prepare_nq_samples = prepare_nq_samples

    def __next__(self):
        lines = list(itertools.islice(self.data, self.chunksize))
        if lines:
            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as p:
                obj = p.map(self.prepare_nq_samples, lines)
            return obj

        self.close()
        raise StopIteration


class NQDataset(Dataset):
    """
    A simple dataset based on abstract torch.utils.data.Dataset class
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def train_collate_fn(samples: List[Dict]) -> Tuple[List[torch.Tensor], List[torch.LongTensor]]:
    """
    A simple collate function that uses the concept dynamic padding. It's used in a training
    dataloader

    :param samples: training samples that form a batch
    :return: inputs: list containing input_ids and attention_masks as torch.Tensors
             labels: list containing start_positions and end_positions and class_labels
             as torch.LongTensors
    """

    # obtaining input tokens

    max_len = max([len(sample['input_ids']) for sample in samples])

    tokens = np.zeros((len(samples), max_len), dtype=np.int64)
    for i, sample in enumerate(samples):
        row = sample['input_ids']
        tokens[i, :len(row)] = row

    attention_mask = tokens > 0

    inputs = [torch.from_numpy(tokens),
              torch.from_numpy(attention_mask)]

    # obtaining output labels

    start_positions = np.array([sample['start_position'] for sample in samples])
    end_positions = np.array([sample['end_position'] for sample in samples])

    start_positions = np.where(start_positions >= max_len, -1, start_positions)
    end_positions = np.where(end_positions >= max_len, -1, end_positions)

    class_labels = [sample['class_label'] for sample in samples]

    labels = [torch.LongTensor(start_positions),
              torch.LongTensor(end_positions),
              torch.LongTensor(class_labels)]

    return inputs, labels


def eval_collate_fn(samples: List[Dict]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    A simple collate function that uses the concept dynamic padding. It's used in a eval
    dataloader

    :param samples: eval samples that form a batch
    :return: inputs: list containing input_ids and attention_masks as torch.Tensors
             samples: list containing original eval samples that form a batch
    """

    # obtaining input tokens

    max_len = max([len(sample['input_ids']) for sample in samples])

    tokens = np.zeros((len(samples), max_len), dtype=np.int64)
    for i, sample in enumerate(samples):
        row = sample['input_ids']
        tokens[i, :len(row)] = row

    attention_mask = tokens > 0

    inputs = [torch.from_numpy(tokens),
              torch.from_numpy(attention_mask)]

    return inputs, samples
