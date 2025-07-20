import os
import json
import torch
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
import evaluate
import numpy as np
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from clippyadagrad import ClippyAdagrad
from torch.optim import AdamW
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Device Setup
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Define custom Trainer
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Use the precomputed class weights
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def get_class_weights(train_dataset, num_classes):
    """
    Compute class weights based on the training dataset.

    Args:
        train_dataset: The training dataset.
        num_classes (int): The number of classes.

    Returns:
        torch.Tensor: Tensor containing class weights.
    """
    # Extract labels from the dataset
    labels = []
    for i in range(len(train_dataset)):
        labels.append(train_dataset[i]['labels'])
    labels = np.array(labels)

    # Compute class weights
    class_weights_np = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=labels
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)
    return class_weights


def compute_metrics(eval_pred):
    """
    Compute accuracy metrics for evaluation.

    Args:
        eval_pred (tuple): A tuple containing predictions and labels.

    Returns:
        dict: A dictionary containing the accuracy score.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)

    return {
        'accuracy': report['accuracy'],
        'f1': report['macro avg']['f1-score'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall']
    }

def set_seed(seed=3):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def append_hyperparams_to_summary(hyperparams, hyperparams_path, experiment_name):
    """
    Append hyperparameters to a hyperparameters CSV file.

    Args:
        hyperparams (dict): Dictionary of hyperparameters.
        hyperparams_path (str): Path to the hyperparameters CSV file.
        experiment_name (str): Name of the experiment.
    """
    # Add timestamp and experiment name
    import datetime
    hyperparams['timestamp'] = datetime.datetime.now().isoformat()
    hyperparams['experiment'] = experiment_name

    # Convert to DataFrame
    df_hyperparams = pd.DataFrame([hyperparams])

    # Append to CSV
    if not os.path.exists(hyperparams_path):
        df_hyperparams.to_csv(hyperparams_path, index=False)
    else:
        df_hyperparams.to_csv(hyperparams_path, mode='a', header=False, index=False)



def train_model(pretrained_model, train_dataset, val_dataset, output_dir="./model", experiment_name="experiment"):
    """
    Train a model using the provided training and validation datasets.

    Args:
        model_name_or_path (str): Path to the pre-trained model or model identifier from HuggingFace.
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        output_dir (str): Directory to save the trained model.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize W&B
    wandb.init(project="model-training", name=experiment_name, dir=output_dir)

    # Set random seed for reproducibility
    set_seed(3)

    # model = AutoModelForSequenceClassification.from_pretrained("./model", num_labels=3)

    # Use a smaller model to reduce overfitting
    # model_name_or_path = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model.to(device)  # Move model to the appropriate device

    # Compute class weights
    num_classes = model.config.num_labels
    class_weights = get_class_weights(train_dataset, num_classes).to(device)

    # Apply drop rate
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.2  # dropout probability

    # optimizer = ClippyAdagrad(model.parameters(), lr=3e-5)
    optimizer = ClippyAdagrad([
        {'params': model.bert.encoder.layer[:6].parameters(), 'lr': 1e-5},
        {'params': model.bert.encoder.layer[6:].parameters(), 'lr': 2e-5},
        {'params': model.bert.pooler.parameters(), 'lr': 2e-5},
        {'params': model.classifier.parameters(), 'lr': 3e-5},
    ], lr=3e-5)

    # # Define optimizer with layer-wise learning rates
    # optimizer = AdamW([
    #     {'params': model.bert.encoder.layer[:6].parameters(), 'lr': 1e-5},
    #     {'params': model.bert.encoder.layer[6:].parameters(), 'lr': 2e-5},
    #     {'params': model.bert.pooler.parameters(), 'lr': 2e-5},
    #     {'params': model.classifier.parameters(), 'lr': 3e-5},
    # ], lr=3e-5)

    training_args = TrainingArguments(
        output_dir=output_dir,  # The output directory
        logging_steps=500,      # Log every 1000 steps
        save_strategy="epoch",  # Save checkpoint at the end of each epoch
        num_train_epochs=15,    # Number of training epochs
        per_device_train_batch_size=12,  # Batch size per device during training
        per_device_eval_batch_size=12,   # Batch size for evaluation
        warmup_steps=500,       # Number of warmup steps for learning rate scheduler /100
        learning_rate=5e-5,     # Learning rate 3e-5
        weight_decay=0.01,      # Strength of weight decay
        eval_strategy="epoch",  # Evaluation strategy to adopt during training
        save_total_limit=6,     # Limit the total amount of checkpoints
        logging_dir=os.path.join(output_dir, 'logs'),  # Directory for TensorBoard logs within experiment directory
        load_best_model_at_end=True,  # Load the best model when finished training
        metric_for_best_model='eval_f1',  # Use f1 score to evaluate the best model
        greater_is_better=True, # Whether the `metric_for_best_model` should be maximized or not
        gradient_accumulation_steps=3,      # Adjust based on GPU memory
        report_to=["wandb"],  # Enable W&B reporting
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    # Verify model device placement
    print(f"Trainer model is on device: {next(trainer.model.parameters()).device}")


    trainer.train()
    trainer.save_model(output_dir)
    eval_results = trainer.evaluate()
    print(eval_results)


    # Log evaluation metrics to W&B
    wandb.log(eval_results)

    # Save evaluation results to a JSON file
    metrics_path = os.path.join(output_dir, 'eval_results.json')
    with open(metrics_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    print(f"Evaluation results saved to {metrics_path}")


    # Save hyperparameters
    hyperparams = {
        'learning_rate': training_args.learning_rate,
        'batch_size': training_args.per_device_train_batch_size,
        'num_epochs': training_args.num_train_epochs,
        'weight_decay': training_args.weight_decay,
        'gradient_accumulation_steps': training_args.gradient_accumulation_steps,
        'optimizer': 'ClippyAdagrad',
        'dropout_rate': 0.2
    }
    hyperparams_csv_path = os.path.join(output_dir, 'hyperparameters.csv')
    append_hyperparams_to_summary(hyperparams, hyperparams_csv_path, experiment_name)
    print(f"Hyperparameters saved to {hyperparams_csv_path}")


    # Plotting functions
    plot_loss_curves_from_trainer(trainer, output_dir)
    plot_metrics_curves_from_trainer(trainer, output_dir)
    plot_confusion_matrix(trainer, val_dataset, output_dir)
    plot_classification_report(trainer, val_dataset, output_dir)

    # Finish W&B run
    wandb.finish()

def evaluate_model(model_name_or_path, val_dataset):
    """
    Evaluate a model using the provided validation dataset.

    Args:
        model_name_or_path (str): Path to the pre-trained model or model identifier from HuggingFace.
        val_dataset (Dataset): The validation dataset.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = AutoModelForSequenceClassification.from_pretrained("./model", num_labels=3)
    model.to(device)  # Move model to the appropriate device

    training_args = TrainingArguments(
        output_dir=os.path.join(model_name_or_path, "evaluation_results"),
        per_device_eval_batch_size=12,
        eval_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset
    )

    # Verify model device placement
    print(f"Evaluator model is on device: {next(trainer.model.parameters()).device}")

    result = trainer.evaluate()
    print(result)

    # Save evaluation metrics to summary.csv
    summary_csv_path = os.path.join(model_name_or_path, 'summary.csv')
    append_metrics_to_summary(result, summary_csv_path)
    print(f"Evaluation results appended to {summary_csv_path}")

def test_model(model_name_or_path, test_dataset):
    """
    Test a model using the provided test dataset and return predictions.

    Args:
        model_name_or_path (str): Path to the pre-trained model or model identifier from HuggingFace.
        test_dataset (Dataset): The test dataset.

    Returns:
        np.ndarray: An array of predicted labels.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = AutoModelForSequenceClassification.from_pretrained("./model", num_labels=3)
    model.to(device)  # Move model to the appropriate device
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    training_args = TrainingArguments(
        output_dir=os.path.join(model_name_or_path, "test_results"),
        per_device_eval_batch_size=12,
        eval_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    # Verify model device placement
    print(f"Tester model is on device: {next(trainer.model.parameters()).device}")

    predictions = trainer.predict(test_dataset)

    preds = np.argmax(predictions.predictions, axis=-1)
    return preds


# Plotting functions
def plot_loss_curves_from_trainer(trainer, output_dir):
    """Plots training and validation loss curves."""
    log_history = trainer.state.log_history
    df = pd.DataFrame(log_history)

    # Filter for 'loss' and 'eval_loss'
    df_loss = df.dropna(subset=["loss"])
    df_eval = df.dropna(subset=["eval_loss"])

    plt.figure(figsize=(8,6))
    plt.plot(df_loss['epoch'], df_loss['loss'], label='Training Loss')
    plt.plot(df_eval['epoch'], df_eval['eval_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Loss curve saved to {plot_path}")

def plot_metrics_curves_from_trainer(trainer, output_dir):
    """Plots validation accuracy and F1-score over epochs."""
    log_history = trainer.state.log_history
    df = pd.DataFrame(log_history)

    # Filter for evaluation logs
    df_metrics = df.dropna(subset=['eval_loss'])

    epochs = df_metrics['epoch']
    accuracy = df_metrics['eval_accuracy']
    f1 = df_metrics['eval_f1']

    plt.figure(figsize=(8,6))
    plt.plot(epochs, accuracy, label='Validation Accuracy')
    plt.plot(epochs, f1, label='Validation F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Accuracy and F1-Score Over Epochs')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'metrics_curve.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Metrics curve saved to {plot_path}")


def plot_confusion_matrix(trainer, val_dataset, output_dir):
    """Plots the confusion matrix for the validation dataset."""
    # Get predictions
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, preds)
    labels_list = ['Class 0', 'Class 1', 'Class 2']

    # Plot
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_list, yticklabels=labels_list)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Confusion matrix saved to {plot_path}")


def plot_classification_report(trainer, val_dataset, output_dir):
    """Plots the classification report as a heatmap."""
    # Get predictions
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    # Generate classification report
    labels_list = ['Class 0', 'Class 1', 'Class 2']
    report = classification_report(true_labels, preds, target_names=labels_list, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    # Select only classes (exclude accuracy, macro avg, weighted avg)
    df_report = df_report.iloc[:-3, :-1]  # Exclude support column

    # Plot
    plt.figure(figsize=(8,6))
    sns.heatmap(df_report, annot=True, cmap='Blues')
    plt.title('Classification Report')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'classification_report.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Classification report saved to {plot_path}")

def append_metrics_to_summary(metrics, summary_path):
    """
    Append evaluation metrics to a summary CSV file.

    Args:
        metrics (dict): Dictionary of evaluation metrics.
        summary_path (str): Path to the summary CSV file.
    """
    # Add timestamp and possibly experiment parameters
    import datetime
    metrics['timestamp'] = datetime.datetime.now().isoformat()

    # Convert to DataFrame
    df_metrics = pd.DataFrame([metrics])

    # Append to CSV
    if not os.path.exists(summary_path):
        df_metrics.to_csv(summary_path, index=False)
    else:
        df_metrics.to_csv(summary_path, mode='a', header=False, index=False)
