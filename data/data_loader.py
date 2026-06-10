import torch
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd

converter = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

def split_input(df: pd.DataFrame) -> torch.Tensor:
    pass


def split_groundtruth(df: pd.DataFrame) -> torch.Tensor:
    truth_frame = df["brent_crude_usd"]

    return torch.Tensor(truth_frame)

if __name__ == "__main__":
    