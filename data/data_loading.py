import pandas as pd
import math


def load_data(data_path, splitted=True):
    # Load CSV
    table = pd.read_csv(data_path, index_col=0).values

    if splitted:
        # Compute train- and validation index
        train_idx = math.ceil(table[:, :5].shape[0] * 0.75)
        val_idx = math.floor(table[train_idx:].shape[0] / 2)

        # Split data into train-, validation- and test-split
        train_split = table[:train_idx]
        val_split = table[train_idx:train_idx+val_idx]
        test_split = table[train_idx+val_idx:]

        # Return splits
        return (
            (train_split[:, :5], train_split[:, 5:]),
            (val_split[:, :5], val_split[:, 5:]),
            (test_split[:, :5], test_split[:, 5:])
        )
    else:
        return table[:, :5], table[:, 5:]
