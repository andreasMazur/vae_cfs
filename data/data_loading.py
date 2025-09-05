import pandas as pd
import math

# Column names to integer
COL_E_MODULUS = 0
COL_THICKNESS = 1
COL_BENDING_ANGLE = 2
COL_BENDING_RADIUS = 3
COL_W1 = 4


def load_data(data_path, split=True):
    """Load data from CSV file and normalize it using the z-score.

    Parameters
    ----------
    data_path: str
        Path to the CSV file containing the data.
    split: bool
        Whether to split the data into training, validation, and test sets.
    """
    # Load CSV
    table = pd.read_csv(data_path, index_col=0).values

    if split:
        # Compute train- and validation index
        train_idx = math.ceil(table[:, :COL_W1].shape[0] * 0.75)
        val_idx = math.floor(table[train_idx:].shape[0] / 2)

        # Split data into train-, validation- and test-split
        train_split = table[:train_idx]
        val_split = table[train_idx:train_idx+val_idx]
        test_split = table[train_idx+val_idx:]

        # Return splits
        return (
            (train_split[:, :COL_W1], train_split[:, COL_W1:]),
            (val_split[:, :COL_W1], val_split[:, COL_W1:]),
            (test_split[:, :COL_W1], test_split[:, COL_W1:])
        )
    else:
        return table[:, :COL_W1], table[:, COL_W1:]
