# baseline.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_data():
    """
    Load and preprocess the training and validation data from CSV files.

    The function reads 'train.csv' and 'valid.csv', combines the 'claim' and 'text' columns,
    and returns the inputs and labels for training and validation.

    Returns:
        tuple: A tuple containing:
            - train_inputs (list): List of combined claim and text for training.
            - train_labels (list): List of labels for training data.
            - val_inputs (list): List of combined claim and text for validation.
            - val_labels (list): List of labels for validation data.
    """
    train_df = pd.read_csv('train.csv')
    train_texts = train_df['text'].astype(str).tolist()
    train_claims = train_df['claim'].astype(str).tolist()
    train_labels = train_df['label'].astype(int).tolist()

    train_inputs = [claim + ' ' + text for claim, text in zip(train_claims, train_texts)]

    val_df = pd.read_csv('valid.csv')
    val_texts = val_df['text'].astype(str).tolist()
    val_claims = val_df['claim'].astype(str).tolist()
    val_labels = val_df['label'].astype(int).tolist()

    val_inputs = [claim + ' ' + text for claim, text in zip(val_claims, val_texts)]

    return train_inputs, train_labels, val_inputs, val_labels

def train_baseline(train_inputs, train_labels, experiment_dir):
    """
    Train a baseline Multinomial Naive Bayes model using TF-IDF features.

    Args:
        train_inputs (list): List of combined claim and text for training.
        train_labels (list): List of labels for training data.
        experiment_dir (str): Directory to save the trained model and vectorizer.

    Returns:
        tuple: A tuple containing:
            - clf (MultinomialNB): Trained Multinomial Naive Bayes model.
            - vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
    """
    os.makedirs(experiment_dir, exist_ok=True)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train = vectorizer.fit_transform(train_inputs)

    clf = MultinomialNB()
    clf.fit(X_train, train_labels)

    clf_path = os.path.join(experiment_dir, 'baseline_model.pkl')
    vectorizer_path = os.path.join(experiment_dir, 'vectorizer.pkl')
    joblib.dump(clf, clf_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Baseline model saved to {clf_path}")
    print(f"TF-IDF vectorizer saved to {vectorizer_path}")

    return clf, vectorizer

def evaluate_baseline(clf, vectorizer, val_inputs, val_labels, output_dir):
    """
    Evaluate the baseline model on the validation data and generate plots.

    Args:
        clf (MultinomialNB): Trained Multinomial Naive Bayes model.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        val_inputs (list): List of combined claim and text for validation.
        val_labels (list): List of labels for validation data.
        output_dir (str): Directory to save the evaluation plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_val = vectorizer.transform(val_inputs)
    val_preds = clf.predict(X_val)

    # Compute metrics
    accuracy = accuracy_score(val_labels, val_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro')
    print(f'Baseline Model Accuracy: {accuracy:.4f}')
    print(f'Baseline Model Precision: {precision:.4f}')
    print(f'Baseline Model Recall: {recall:.4f}')
    print(f'Baseline Model F1 Score: {f1:.4f}')
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds))

    # Save metrics to summary.csv
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    summary_csv_path = os.path.join(output_dir, 'summary.csv')
    append_metrics_to_summary(metrics, summary_csv_path, experiment_name=os.path.basenameexperiment_dir())
    print(f"Summary metrics appended to {summary_csv_path}")

    # Generate and save plots
    plot_confusion_matrix(val_labels, val_preds, output_dir)
    plot_classification_report(val_labels, val_preds, output_dir)

def plot_confusion_matrix(true_labels, preds, output_dir):
    """
    Plots the confusion matrix for the validation dataset.

    Args:
        true_labels (list or array): True labels of the validation data.
        preds (list or array): Predicted labels by the model.
        output_dir (str): Directory to save the confusion matrix plot.
    """
    cm = confusion_matrix(true_labels, preds)
    labels_list = ['Class 0', 'Class 1', 'Class 2']

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_list, yticklabels=labels_list)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Baseline Model Confusion Matrix')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'baseline_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Confusion matrix saved to {plot_path}")

def plot_classification_report(true_labels, preds, output_dir):
    """
    Plots the classification report as a heatmap.

    Args:
        true_labels (list or array): True labels of the validation data.
        preds (list or array): Predicted labels by the model.
        output_dir (str): Directory to save the classification report plot.
    """
    labels_list = ['Class 0', 'Class 1', 'Class 2']
    report = classification_report(true_labels, preds, target_names=labels_list, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    # Select only classes (exclude accuracy, macro avg, weighted avg)
    df_report = df_report.iloc[:-3, :-1]  # Exclude support column

    plt.figure(figsize=(8,6))
    sns.heatmap(df_report, annot=True, cmap='Blues')
    plt.title('Baseline Model Classification Report')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'baseline_classification_report.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Classification report saved to {plot_path}")

def append_metrics_to_summary(metrics, summary_path, experiment_name):
    """
    Append evaluation metrics to a summary CSV file.

    Args:
        metrics (dict): Dictionary of evaluation metrics.
        summary_path (str): Path to the summary CSV file.
        experiment_name (str): Name of the experiment.
    """
    # Add timestamp and experiment name
    import datetime
    metrics['timestamp'] = datetime.datetime.now().isoformat()
    metrics['experiment'] = experiment_name

    # Convert to DataFrame
    df_metrics = pd.DataFrame([metrics])

    # Append to CSV
    if not os.path.exists(summary_path):
        df_metrics.to_csv(summary_path, index=False)
    else:
        df_metrics.to_csv(summary_path, mode='a', header=False, index=False)

def test_baseline(clf, vectorizer, experiment_dir):
    """
    Test the baseline model on the test data and save the predictions to a CSV file.

    The function reads 'test.csv', combines the 'claim' and 'text' columns,
    makes predictions using the trained model, and saves the results to 'baseline_output.csv'.

    Args:
        clf (MultinomialNB): Trained Multinomial Naive Bayes model.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        experiment_dir (str): Directory to save the test predictions.
    """
    test_df = pd.read_csv('test.csv')
    test_texts = test_df['text'].astype(str).tolist()
    test_claims = test_df['claim'].astype(str).tolist()
    test_ids = test_df['id'].tolist()

    test_inputs = [claim + ' ' + text for claim, text in zip(test_claims, test_texts)]

    X_test = vectorizer.transform(test_inputs)
    test_preds = clf.predict(X_test)

    output = {'id': test_ids, 'rating': test_preds}
    output_df = pd.DataFrame(output)
    output_csv_path = os.path.join(experiment_dir, 'baseline_output.csv')
    output_df.to_csv(output_csv_path, index=False)
    print(f"Baseline predictions saved to '{output_csv_path}'")

def main():
    """
    Main function to load data, train, evaluate, and test the baseline model.

    Steps:
        - Load training and validation data.
        - Train the baseline Multinomial Naive Bayes model.
        - Evaluate the model on validation data.
        - Test the model on test data and save predictions.
    """

    import argparse

    parser = argparse.ArgumentParser(description='Baseline Model Training and Evaluation')
    parser.add_argument('--experiment_name', type=str, required=True, help='Unique name for the experiment.')
    args = parser.parse_args()

    # Create a unique output directory for the experiment
    experiment_dir = os.path.join('./baseline', args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    print("Loading data...")
    train_inputs, train_labels, val_inputs, val_labels = load_data()

    print("Training baseline model...")
    clf, vectorizer = train_baseline(train_inputs, train_labels, experiment_dir)

    print("Evaluating baseline model...")
    evaluate_baseline(clf, vectorizer, val_inputs, val_labels, experiment_dir)

    print("Testing baseline model...")
    test_baseline(clf, vectorizer, experiment_dir)

if __name__ == '__main__':
    main()
