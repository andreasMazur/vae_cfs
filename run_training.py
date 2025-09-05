from compute_cf.experiment_series import train_on_partial_data_wrapper


# TODO: Set these paths
LOGGING_DIR = ""
REPOSITORY_DIR = ""

# TODO: Set number of experiment repetitions
REPETITIONS = 100


if __name__ == "__main__":
    train_on_partial_data_wrapper(
        data_path=f"{REPOSITORY_DIR}/data/punch_bending_data.csv",
        logging_dir=LOGGING_DIR,
        repetitions=REPETITIONS
    )
