# main.py

import argparse
import os
from data_processing import compose_data, process_data_for_training, process_data_for_testing
from model_utils import train_model, evaluate_model, test_model
from transformers import AutoTokenizer
import pandas as pd

def main():
    """
    Main function to handle model training, evaluation, and testing.

    This function parses command-line arguments to determine the operation mode:
    - Compose data from raw files.
    - Train the model.
    - Evaluate the model.
    - Test the model.

    It then calls the appropriate functions from the data_processing and model_utils modules.
    """
    parser = argparse.ArgumentParser(description='Model Training and Evaluation')
    parser.add_argument('--compose', action='store_true', help='Compose data from raw files.')
    parser.add_argument('--train', action='store_true', help='Train the model.')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model.')
    parser.add_argument('--test', action='store_true', help='Test the model.')
    parser.add_argument('--experiment_name', type=str, default='experiment', help='Unique name for the experiment.')
    parser.add_argument('--model_path', type=str, default='./model', help='Base directory to save experiments.')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased', help='Pre-trained model identifier or path.')
    args = parser.parse_args()

    # Create a unique output directory for the experiment
    experiment_dir = os.path.join(args.model_path, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    if args.compose:
        print("Composing data...")
        compose_data()

    if args.train or args.evaluate:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        print("Processing data for training/evaluation...")
        train_dataset, val_dataset = process_data_for_training(tokenizer)

    if args.train:
        print("Training the model...")
        train_model(
            pretrained_model=args.pretrained_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=experiment_dir,
            experiment_name=args.experiment_name
        )

    if args.evaluate:
        print("Evaluating the model...")
        evaluate_model(
            model_name_or_path=experiment_dir,  # Load model from experiment-specific directory
            val_dataset=val_dataset
        )

    if args.test:
        print("Processing data for testing...")
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        test_dataset, ids = process_data_for_testing(tokenizer)
        print("Testing the model...")
        preds = test_model(
            model_name_or_path=experiment_dir,
            test_dataset=test_dataset
        )

        # Save the output
        print("Saving test results...")
        output = {"id": ids, "rating": preds}
        DF = pd.DataFrame.from_dict(output)
        output_csv_path = os.path.join(experiment_dir, "test_output.csv")
        DF.to_csv(output_csv_path, index=False)
        print(f"Test results saved to {output_csv_path}")

if __name__ == '__main__':
    main()
