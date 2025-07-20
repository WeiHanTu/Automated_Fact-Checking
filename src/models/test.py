from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset


class FactDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


df = pd.read_csv("./test.csv")
texts = df["text"]
claims = df["claim"]
ids = df["id"]

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

encoding = {"input_ids": input_ids}
test_dataset = FactDataset(encoding)


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
)

prediction = trainer.predict(test_dataset)
print(prediction)

index = []
ans = []
for i in range(len(prediction.predictions)):
    index.append(i)
    logit = np.array(prediction.predictions[i])
    ans.append(np.argmax(logit))

output = {"id": ids, "rating": ans}

DF = pd.DataFrame.from_dict(output)
DF.to_csv("./output.csv", index=False)
