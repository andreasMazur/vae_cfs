from evaluation.baseline import baseline_performance
from evaluation.compare_outcome import compare_outcome_targets, compare_outcome_closest_nn
from evaluation.mrad_eval import mrad_eval_target, mrad_eval_closest_nn, FEATURE_NAMES
from evaluation.permutation_feature_importance import pfi_plots


def evaluation_wrapper(data_path, logs_dir, data_distance_path):
    ks = [4681 * i for i in range(8, 15)][::-1]
    for max_reps in [125]:
        for feature in range(4):
            pfi_plots(
                data_path=data_path,
                logs_dir=logs_dir,
                ks=ks,
                feature=feature,
                file_name=f"pfi_mr{max_reps}_{FEATURE_NAMES[feature]}",
                max_reps=max_reps
            )
            mrad_eval_target(
                data_path=data_path,
                logs_dir=logs_dir,
                data_distance_path=data_distance_path,
                ks=ks,
                feature=feature,
                file_name=f"cfs_target_mr{max_reps}_{FEATURE_NAMES[feature]}",
                max_reps=max_reps
            )
            mrad_eval_closest_nn(
                data_path=data_path,
                logs_dir=logs_dir,
                data_distance_path=data_distance_path,
                ks=ks,
                feature=feature,
                file_name=f"cfs_nn_mr{max_reps}_{FEATURE_NAMES[feature]}",
                max_reps=max_reps
            )

        # Outcome evaluation
        baseline_performance(
            logs_dir=logs_dir,
            ks=ks,
            file_name=f"baseline_performance_mr{max_reps}",
            max_reps=max_reps
        )
        compare_outcome_targets(
            logs_dir=logs_dir, ks=ks, file_name=f"cf_performance_mr{max_reps}", max_reps=max_reps
        )
        compare_outcome_closest_nn(
            data_path=data_path,
            logs_dir=logs_dir,
            ks=ks,
            file_name=f"cf_nn_y_distance_mr{max_reps}",
            max_reps=max_reps
        )
