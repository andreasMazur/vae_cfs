import pandas as pd
import math

# Decide whether to use clamping here
# None: No clamping filter
# 0: Only clamping = 0
# 1: Only clamping = 1
USE_CLAMPING_FILTER = 0
assert USE_CLAMPING_FILTER in [None, 0, 1], "USE_CLAMPING_FILTER must be None, 0 or 1."
if USE_CLAMPING_FILTER is None:
    COL_E_MODULUS = 0
    COL_THICKNESS = 1
    COL_CLAMPING = 2
    COL_BENDING_ANGLE = 3
    COL_BENDING_RADIUS = 4
    COL_W1 = 5
else:
    COL_E_MODULUS = 0
    COL_THICKNESS = 1
    COL_CLAMPING = None
    COL_BENDING_ANGLE = 2
    COL_BENDING_RADIUS = 3
    COL_W1 = 4


def load_data(data_path, splitted=True):
    # Load CSV
    table = pd.read_csv(data_path, index_col=0).values

    # Filter clamping values if wanted
    if USE_CLAMPING_FILTER == 0:
        table = table[table[:, 2] == 0]
        table = table[:, [0, 1, 3, 4, 5]]
    elif USE_CLAMPING_FILTER == 1:
        table = table[table[:, 2] == 1]
        table = table[:, [0, 1, 3, 4, 5]]

    if splitted:
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
