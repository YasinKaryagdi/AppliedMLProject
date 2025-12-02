from datasets import load_dataset
import datasets
from transformers import Trainer
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments
from datasets import DatasetDict
import numpy as np
import os
import matplotlib.pyplot as plt
import evaluate

base = "../"

#fix pathing function
def rel_path(path: str) -> str:
    return os.path.join(base, path)

def fix_paths(set):
    set["image_path"] = os.path.join(base, set["image_path"].lstrip("/"))
    return set

#load dataset through csv file
dataset = load_dataset(
    "csv",
    data_files= "../train_images.csv")

#fix image paths
dataset["train"] = dataset["train"].map(fix_paths)
#cast image on dataset
dataset = dataset.cast_column("image_path", datasets.Image())
#rename columns for model compatibility
dataset = dataset.rename_column("image_path", "image")
dataset = dataset.rename_column("label", "labels")
#convert to train and test split
dataset = dataset["train"].train_test_split(test_size=0.15, seed=8)
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

model_name = "microsoft/resnet-50"
processor = AutoImageProcessor.from_pretrained(model_name)
classes = np.load(os.path.join("../class_names.npy"), allow_pickle=True)
num_classes = len(classes.item())

model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)

def transform(example):
    inputs = processor(example["image"], return_tensors="pt")
    example["pixel_values"] = inputs["pixel_values"][0]
    return example

dataset = dataset.with_transform(transform)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# accuracy = evaluate.load("accuracy")

# def compute_metrics(pred):
#     logits, labels = pred
#     preds = logits.argmax(-1)
#     return accuracy.compute(predictions=preds, references=labels)

# training_args = TrainingArguments(
#     output_dir="./resnet50-feathers",
#     num_train_epochs=5,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     save_strategy="epoch",
#     learning_rate=1e-4,
#     weight_decay=0.01,
#     fp16=True,
#     logging_steps=20,
#     remove_unused_columns=False,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["validation"],
# )
# print("Starting training...")

# trainer.train()
