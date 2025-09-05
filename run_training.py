from compute_cf.experiment_series import train_on_partial_data_wrapper


# TODO: Set these paths
DATASET_PATH = ""
LOGGING_DIR = ""

# TODO: Set number of experiment repetitions
REPETITIONS = 100


if __name__ == "__main__":
    train_on_partial_data_wrapper(
        data_path=DATASET_PATH,
        logging_dir=LOGGING_DIR,
        repetitions=REPETITIONS
    )
