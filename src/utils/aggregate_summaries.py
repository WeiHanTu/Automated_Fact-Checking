# aggregate_summaries.py

import os
import pandas as pd

def aggregate_summaries(model_base_dir="./model"):
    """
    Aggregates all summary.csv and hyperparameters.csv files from different experiments into a master summary.

    Args:
        model_base_dir (str): Base directory where all experiments are saved.
    """
    master_summary_metrics = []
    master_summary_hyperparams = []

    for experiment in os.listdir(model_base_dir):
        experiment_dir = os.path.join(model_base_dir, experiment)
        if os.path.isdir(experiment_dir):
            # Metrics
            summary_path = os.path.join(experiment_dir, 'summary.csv')
            if os.path.isfile(summary_path):
                df_metrics = pd.read_csv(summary_path)
                master_summary_metrics.append(df_metrics)

            # Hyperparameters
            hyperparams_path = os.path.join(experiment_dir, 'hyperparameters.csv')
            if os.path.isfile(hyperparams_path):
                df_hyperparams = pd.read_csv(hyperparams_path)
                master_summary_hyperparams.append(df_hyperparams)

    if master_summary_metrics:
        master_metrics_df = pd.concat(master_summary_metrics, ignore_index=True)
        master_metrics_path = os.path.join(model_base_dir, 'master_summary_metrics.csv')
        master_metrics_df.to_csv(master_metrics_path, index=False)
        print(f"Master metrics summary saved to {master_metrics_path}")
    else:
        print("No metrics summaries found.")

    if master_summary_hyperparams:
        master_hyperparams_df = pd.concat(master_summary_hyperparams, ignore_index=True)
        master_hyperparams_path = os.path.join(model_base_dir, 'master_summary_hyperparams.csv')
        master_hyperparams_df.to_csv(master_hyperparams_path, index=False)
        print(f"Master hyperparameters summary saved to {master_hyperparams_path}")
    else:
        print("No hyperparameters summaries found.")

if __name__ == "__main__":
    aggregate_summaries()
