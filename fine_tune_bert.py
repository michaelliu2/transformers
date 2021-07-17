import numpy as np
import os

from datasets import load_dataset
from datasets import load_metric

from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer

def compute_metrics():
    metric = load_metric('accuracy')
    def func(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    return func


def tokenize_function():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    return func


if os.environ["MODE"] == "weights":
    pretrained_model = "tune_weights"
    stage = "tuned_weights"
    eval_steps = 250
    train_examples = 1000
elif os.environ["MODE"] == "tune":
    pretrained_model = "tuned_weights"
    stage = "tuned_bert"
    eval_steps = 1000
    train_examples = 10000
else:
    raise Exception("Set MODE env variable")


raw_datasets = load_dataset("imdb")
tokenized_datasets = raw_datasets.map(tokenize_function(), batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(train_examples))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(2000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=2)
training_args = TrainingArguments("bert-fine-tuning", evaluation_strategy="steps", eval_steps=eval_steps, save_strategy="epoch", num_train_epochs=8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics()
)

trainer.train()

model.save_pretrained(stage)
