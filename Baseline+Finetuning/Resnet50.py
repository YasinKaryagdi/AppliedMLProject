from datasets import load_dataset
import datasets
from transformers import Trainer
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments
from datasets import DatasetDict
import numpy as np
import os
import matplotlib.pyplot as plt
base = "./AppliedMLProject/aml-2025-feathers-in-focus/"
#load dataset through csv file
data = load_dataset(
    "csv",
    data_files=base + "train_images.csv"
)
print(dataset)
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
plt.imshow(dataset["train"][0]["image"])

# processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

# model_name = "microsoft/resnet-50"
# processor = AutoImageProcessor.from_pretrained(model_name)
# classes = np.load(os.path.join("./AppliedMLProject/aml-2025-feathers-in-focus/","class_names.npy"), allow_pickle=True)
# num_classes = 200

# model = AutoModelForImageClassification.from_pretrained(
#     model_name,
#     num_labels=num_classes,
#     ignore_mismatched_sizes=True
# )

# def transform(example):
#     inputs = processor(example["image"], return_tensors="pt")
#     example["pixel_values"] = inputs["pixel_values"][0]
#     return example

# dataset = dataset.with_transform(transform)

# # accuracy = evaluate.load("accuracy")

# # def compute_metrics(pred):
# #     logits, labels = pred
# #     preds = logits.argmax(-1)
# #     return accuracy.compute(predictions=preds, references=labels)

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

# trainer.train()