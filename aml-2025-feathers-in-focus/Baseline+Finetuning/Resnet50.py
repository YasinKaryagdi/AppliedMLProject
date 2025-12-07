from datasets import load_dataset
import datasets
from transformers import Trainer
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments
from datasets import DatasetDict
import numpy as np
import os
import matplotlib.pyplot as plt
import evaluate
from PIL import Image
from transformers import default_data_collator

def fix_grayscale(example):
    if example["image"].mode != "RGB":
        example["image"] = example["image"].convert("RGB")
    return example

def fix_labels(example):
    example["labels"] = example["labels"] - 1
    return example

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

def preprocess(example):
    inputs = processor(example["image"], return_tensors="pt")
    # pixel_values shape is [1, C, H, W], take the first
    example["pixel_values"] = inputs["pixel_values"][0]
    return example

dataset['train'] = dataset["train"].map(fix_grayscale)
dataset['validation'] = dataset["validation"].map(fix_grayscale)

dataset["train"] = dataset["train"].map(preprocess)
dataset["validation"] = dataset["validation"].map(preprocess)

dataset["train"] = dataset["train"].remove_columns("image")
dataset["validation"] = dataset["validation"].remove_columns("image")

dataset["train"] = dataset["train"].map(fix_labels)
dataset["validation"] = dataset["validation"].map(fix_labels)

model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)

def transform(example):
    inputs = processor(example["image"], return_tensors="pt")
    example["pixel_values"] = inputs["pixel_values"][0]
    return example

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="../resnet50-feathers",
    num_train_epochs=5,
    remove_unused_columns=False,
    per_device_train_batch_size=8,  # Larger batch for CPU
    per_device_eval_batch_size=16,   
    dataloader_num_workers=0,  # MUST be 0 on Windows to avoid multiprocessing issues
    eval_strategy="epoch",  
    save_strategy="epoch",
    logging_steps=100,
    use_cpu=True,  # Use CPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")

outcome = trainer.train()
print("Training complete.")
trainer.save_model("../resnet50-finetuned")