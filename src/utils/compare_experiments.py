# compare_experiments.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_master_summary(metrics_path, hyperparams_path):
    """
    Load the master summary CSV files containing metrics and hyperparameters from all experiments.

    Args:
        metrics_path (str): Path to the master summary metrics CSV file.
        hyperparams_path (str): Path to the master summary hyperparameters CSV file.

    Returns:
        tuple: DataFrames containing metrics and hyperparameters.
    """
    if not os.path.exists(metrics_path):
        print(f"Metrics summary file '{metrics_path}' does not exist.")
        return None, None
    if not os.path.exists(hyperparams_path):
        print(f"Hyperparameters summary file '{hyperparams_path}' does not exist.")
        return None, None

    df_metrics = pd.read_csv(metrics_path)
    df_hyperparams = pd.read_csv(hyperparams_path)
    return df_metrics, df_hyperparams

def plot_comparison(df_metrics, metric, save_path):
    """
    Plot a bar chart comparing a specific metric across experiments.

    Args:
        df_metrics (pd.DataFrame): DataFrame containing metrics.
        metric (str): The metric to compare (e.g., 'accuracy', 'f1').
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10,6))
    sns.barplot(x='experiment', y=metric, data=df_metrics, palette='viridis')
    plt.xlabel('Experiment')
    plt.ylabel(metric.capitalize())
    plt.title(f'Comparison of {metric.capitalize()} Across Experiments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"{metric.capitalize()} comparison plot saved to {save_path}")

def plot_hyperparams_comparison(df_hyperparams, hyperparam, save_path):
    """
    Plot a bar chart comparing a specific hyperparameter across experiments.

    Args:
        df_hyperparams (pd.DataFrame): DataFrame containing hyperparameters.
        hyperparam (str): The hyperparameter to compare (e.g., 'learning_rate').
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10,6))
    sns.barplot(x='experiment', y=hyperparam, data=df_hyperparams, palette='magma')
    plt.xlabel('Experiment')
    plt.ylabel(hyperparam.replace('_', ' ').capitalize())
    plt.title(f'Comparison of {hyperparam.replace("_", " ").capitalize()} Across Experiments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"{hyperparam.replace('_', ' ').capitalize()} comparison plot saved to {save_path}")

def main():
    master_metrics_path = os.path.join('./model', 'master_summary_metrics.csv')
    master_hyperparams_path = os.path.join('./model', 'master_summary_hyperparams.csv')

    df_metrics, df_hyperparams = load_master_summary(master_metrics_path, master_hyperparams_path)
    if df_metrics is None or df_hyperparams is None:
        return

    # Metrics to compare
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        save_path = os.path.join('./model', f'comparison_{metric}.png')
        plot_comparison(df_metrics, metric, save_path)

    # Hyperparameters to compare
    hyperparams = ['learning_rate', 'batch_size', 'num_epochs', 'weight_decay', 'gradient_accumulation_steps']
    for hyperparam in hyperparams:
        save_path = os.path.join('./model', f'comparison_{hyperparam}.png')
        plot_hyperparams_comparison(df_hyperparams, hyperparam, save_path)

if __name__ == "__main__":
    main()
