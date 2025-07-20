from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import pandas as pd
import torch
import evaluate
import numpy as np
from torch.utils.data import Dataset


class FactDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)


df = pd.read_csv("./train.csv")
texts = df["text"]
claims = df["claim"]
labels = df["label"]
labels = list(labels)

model = AutoModelForSequenceClassification.from_pretrained("./model", num_labels=3,)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

for i, text in enumerate(texts):
    texts[i] = "[CLS] " + str(claims[i]) + " [SEP] " + str(text)


tokens = [tokenizer.tokenize(text) for text in texts]
for i, token in enumerate(tokens):
    if len(tokens[i]) > 512:
        tokens[i] = tokens[i][:512]
    while len(tokens[i]) < 512:
        tokens[i].append("[PAD]")
    tokens[i][511] = "[SEP]"

input_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]

re_sample = []
for i, label in enumerate(labels):
    if label == 2:
        re_sample.append(input_ids[i])

for input in re_sample:
    input_ids.append(input)
    labels.append(2)

encoding = {"input_ids": input_ids}
train_dataset = FactDataset(encoding, labels)

df = pd.read_csv("./valid.csv")
texts = df["text"]
claims = df["claim"]
labels = df["label"]

for i, text in enumerate(texts):
    texts[i] = "[CLS] " + str(claims[i]) + " [SEP] " + str(text)

tokens = [tokenizer.tokenize(text) for text in texts]
for i, token in enumerate(tokens):
    if len(tokens[i]) > 512:
        tokens[i] = tokens[i][:512]
    while len(tokens[i]) < 512:
        tokens[i].append("[PAD]")
    tokens[i][511] = "[SEP]"
input_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]

encoding = {"input_ids": input_ids}
val_dataset = FactDataset(encoding, labels)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    # metric = evaluate.load("f1")
    # return metric.compute(predictions=predictions, references=labels, average='macro')
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="./results",
    logging_steps=10000,
    save_steps=10000,
    num_train_epochs=3,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    warmup_steps=150,
    learning_rate=3e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# trainer.train()

# trainer.save_model("./model")

result = trainer.evaluate()
print(result)
