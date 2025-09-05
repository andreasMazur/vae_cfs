from run_training import REPOSITORY_DIR
from evaluation.evaluation_wrapper import evaluation_wrapper


# TODO: Set these paths
DATASET_PATH = ""
LOGGING_DIR = ""


if __name__ == "__main__":
    evaluation_wrapper(
        data_path=DATASET_PATH,
        logs_dir=LOGGING_DIR,
        data_distance_path=f"{REPOSITORY_DIR}/data/data_distance.json"
    )
