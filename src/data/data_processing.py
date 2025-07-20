import pandas as pd
import glob
import re
from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def compose_data():
    """
    Composes the training, validation, and test datasets from raw extracted data and saves them as CSV files.

    This function reads data from JSON files ('train.json', 'valid.json', 'test.json') and text files located in the './extract/' directory.
    It processes the data by cleaning and combining the texts and claims, then saves the processed data into 'train.csv', 'valid.csv', and 'test.csv'.

    Steps:
        - Extract IDs and texts from CSV files in the './extract/' directory.
        - Clean and concatenate texts associated with each ID.
        - Process 'train.json', 'valid.json', and 'test.json' to extract claims and labels.
        - Match the IDs with the corresponding texts and claims.
        - Save the processed data into CSV files for training, validation, and testing.
    """
    filenames = glob.glob('./extract/*.csv')
    ids = []
    texts = []
    for filename in filenames:
        id = re.search('/([^/]*)\.csv', filename).group(1)
        ids.append(int(id))
    ids = sorted(ids)

    for id in tqdm(ids):
        df = pd.read_csv(f"./extract/{id}.csv")
        Texts = df['text']
        text = ''
        for Text in Texts:
            text += str(Text).lower() + ' [SEP] '
        text = re.sub('\.', '', text)
        text = re.sub('[^a-z0-9A-Z\[\]]', ' ', text)
        texts.append(text)

    # Process train.json
    claims = []
    labels = []
    df = pd.read_json('./train.json')
    for i in range(len(df['metadata'])):
        claim = df['metadata'][i]['claim']
        claim = claim.lower()
        claim = re.sub('\.', '', claim)
        claim = re.sub('[^a-z0-9]', ' ', claim)
        claims.append(claim)
        label = df['label'][i]['rating']
        labels.append(label)

    end = len(df['metadata'])
    output = {'id': ids[:end], 'text': texts[:end], 'claim': claims, 'label': labels}
    DF = pd.DataFrame.from_dict(output)
    DF.to_csv('train.csv', index=False)

    # Process valid.json
    claims = []
    labels = []
    df = pd.read_json('./valid.json')
    for i in range(len(df['metadata'])):
        claim = df['metadata'][i]['claim']
        claim = claim.lower()
        claim = re.sub('\.', '', claim)
        claim = re.sub('[^a-z0-9]', ' ', claim)
        claims.append(claim)
        label = df['label'][i]['rating']
        labels.append(label)

    end2 = len(df['metadata'])
    output = {'id': ids[end:end + end2], 'text': texts[end:end + end2], 'claim': claims, 'label': labels}
    DF = pd.DataFrame.from_dict(output)
    DF.to_csv('valid.csv', index=False)

    # Process test.json
    claims = []
    df = pd.read_json('./test.json')
    for i in range(len(df['metadata'])):
        claim = df['metadata'][i]['claim']
        claim = claim.lower()
        claim = re.sub('\.', '', claim)
        claim = re.sub('[^a-z0-9]', ' ', claim)
        claims.append(claim)

    output = {'id': ids[end + end2:], 'text': texts[end + end2:], 'claim': claims}
    DF = pd.DataFrame.from_dict(output)
    DF.to_csv('test.csv', index=False)


def process_data_for_training(tokenizer, max_length=512):
    """
    Processes the data for training and validation by tokenizing the input texts and creating datasets.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to encode the text data.
        max_length (int, optional): The maximum sequence length for tokenization. Defaults to 512.

    Returns:
        tuple: A tuple containing the training dataset and validation dataset.
    """

    class FactDataset(Dataset):
        """
        A custom Dataset class for handling tokenized inputs and labels.

        Args:
            encodings (dict): A dictionary containing tokenized inputs.
            labels (list): A list of labels corresponding to the inputs.
        """

        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, index):
            """
            Retrieves an item by index.

            Args:
                index (int): The index of the item to retrieve.

            Returns:
                dict: A dictionary containing the tokenized input and label for the specified index.
            """
            item = {key: val[index].clone().detach() for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[index])
            return item

        def __len__(self):
            """
            Returns the total number of items.

            Returns:
                int: The length of the dataset.
            """
            return len(self.labels)

    # Load train.csv
    df = pd.read_csv("./train.csv")
    # Prepare inputs without manually adding special tokens
    claims = df["claim"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    # Tokenize using tokenizer's built-in methods
    encodings = tokenizer(
        claims,
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    # Resample instances where label == 2 to address class imbalance
    labels_tensor = torch.tensor(labels)
    indices_label_2 = (labels_tensor == 2).nonzero(as_tuple=True)[0]
    resampled_labels = labels_tensor[indices_label_2]

    # Resample and concatenate all encoding fields
    for key in encodings.keys():
        resampled_data = encodings[key][indices_label_2]
        encodings[key] = torch.cat([encodings[key], resampled_data], dim=0)

    # Extend the labels accordingly
    labels.extend(resampled_labels.tolist())

    # Create the training dataset
    train_dataset = FactDataset(encodings, labels)

    # Prepare validation inputs
    df = pd.read_csv("./valid.csv")
    claims = df["claim"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    encodings = tokenizer(
        claims,
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    # Create the validation dataset
    val_dataset = FactDataset(encodings, labels)

    return train_dataset, val_dataset


def process_data_for_testing(tokenizer, max_length=512):
    """
    Processes the data for testing by tokenizing the input texts and creating a test dataset.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to encode the text data.
        max_length (int, optional): The maximum sequence length for tokenization. Defaults to 512.

    Returns:
        tuple: A tuple containing the test dataset and the list of IDs corresponding to each test instance.
    """

    class FactDataset(Dataset):
        """
        A custom Dataset class for handling tokenized inputs for testing.

        Args:
            encodings (dict): A dictionary containing tokenized inputs.
        """

        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, index):
            """
            Retrieves an item by index.

            Args:
                index (int): The index of the item to retrieve.

            Returns:
                dict: A dictionary containing the tokenized input for the specified index.
            """
            item = {key: val[index].clone().detach() for key, val in self.encodings.items()}
            return item

        def __len__(self):
            """
            Returns the total number of items.

            Returns:
                int: The length of the dataset.
            """
            return len(self.encodings['input_ids'])

    # Prepare test inputs
    df = pd.read_csv("./test.csv")
    claims = df["claim"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()
    ids = df["id"].tolist()

    encodings = tokenizer(
        claims,
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    # Create the test dataset
    test_dataset = FactDataset(encodings)

    return test_dataset, ids
