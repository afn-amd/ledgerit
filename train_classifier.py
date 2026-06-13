import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ----------------------------------------
# LOAD DATA
# ----------------------------------------

df = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "dataset", "train.csv")
)

# Remove bad labels if any
valid_labels = ["header", "entry", "c-entry", "info", "remove"]

df = df[df["label"].isin(valid_labels)]

# ----------------------------------------
# LABEL ENCODING
# ----------------------------------------

label2id = {
    "header": 0,
    "entry": 1,
    "c-entry": 2,
    "info": 3,
    "remove": 4
}

id2label = {v:k for k,v in label2id.items()}

df["label_id"] = df["label"].map(label2id)

# ----------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label_id"],
    random_state=42
)

# ----------------------------------------
# TOKENIZER
# ----------------------------------------

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# ----------------------------------------
# DATASETS
# ----------------------------------------

train_dataset = Dataset.from_pandas(
    train_df[["text", "label_id"]]
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label_id"]]
)

train_dataset = train_dataset.rename_column(
    "label_id",
    "labels"
)

test_dataset = test_dataset.rename_column(
    "label_id",
    "labels"
)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# ----------------------------------------
# CLASS WEIGHTS
# ----------------------------------------

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df["label_id"]),
    y=df["label_id"]
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
)

# ----------------------------------------
# MODEL
# ----------------------------------------

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=5,
    id2label=id2label,
    label2id=label2id
)

# ----------------------------------------
# CUSTOM TRAINER
# ----------------------------------------

class WeightedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        labels = inputs.get("labels")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        logits = outputs.get("logits")

        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(model.device)
        )

        loss = loss_fn(
            logits,
            labels
        )

        return (loss, outputs) if return_outputs else loss

# ----------------------------------------
# METRICS
# ----------------------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)

    f1 = f1_score(
        labels,
        predictions,
        average="weighted"
    )

    return {
        "accuracy": acc,
        "f1": f1
    }

# ----------------------------------------
# TRAINING ARGS
# ----------------------------------------

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True
)

# ----------------------------------------
# TRAINER
# ----------------------------------------

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# ----------------------------------------
# TRAIN
# ----------------------------------------

trainer.train()

# ----------------------------------------
# SAVE MODEL
# ----------------------------------------

model.save_pretrained(
    os.path.join(os.path.dirname(__file__), "models", "row_classifier")
)

tokenizer.save_pretrained(
    os.path.join(os.path.dirname(__file__), "models", "row_classifier")
)

print("\nMODEL SAVED")