from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision import datasets, models, transforms
from torchvision.models.efficientnet import efficientnet_b0
from models.modelM3AVG import ModelM3AVG
from models.modelM3MAX import ModelM3MAX
from code.models.random import CNN


from pathlib import Path
import os
import time
import copy
import pickle
from tqdm import tqdm

import numpy as np

from datasets import load_dataset
from CSVDataset import CSVDataset

    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                schedular=None,
                num_epochs=50,
                device="cuda"):
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    since = time.time()
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_acc_history = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
              if phase == 'train':
                  model.train()  # Set model to training mode
              else:
                  model.eval()   # Set model to evaluate mode

              running_loss = 0.0
              running_corrects = 0

              # Iterate over data.
              for inputs, labels in tqdm(dataloaders_dict[phase]):
                  inputs = inputs.to(device)
                  labels = labels.to(device)

                  # zero the parameter gradients
                  optimizer.zero_grad()

                  # forward
                  # track history if only in train
                  with torch.set_grad_enabled(phase == 'train'):
                      outputs = model(inputs)
                      loss = criterion(outputs, labels)

                      _, preds = torch.max(outputs, 1)

                      # backward + optimize only if in training phase
                      if phase == 'train':
                          loss.backward()
                          optimizer.step()

                  # statistics
                  running_loss += loss.item() * inputs.size(0)
                  running_corrects += torch.sum(preds == labels.data)

              epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
              epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)

              print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

              # deep copy the model
              if phase == 'val' and epoch_acc > best_acc:
                  best_acc = epoch_acc
                  best_model_wts = copy.deepcopy(model.state_dict())
              if phase == 'val':
                  val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


cwd = Path.cwd()
gitpath = cwd
dirpath = gitpath / "aml-2025-feathers-in-focus"
train_images_csv = dirpath / "train_images.csv"
train_images_folder = dirpath / "train_images"
image_classes = dirpath / "class_names.npy"


#Defining model and training variables
#use augmented trainingset
use_augmented = True
#model
model_name = "ensem3max"
curr_seed = 0
#possible models: "squeezenet", "resnet50", "resnet152", "efficientnetb0"
# "efficiennetV2s", "SwinV2t"


#Epochs
num_epochs = 20
#feature extraction option (freeze)
feature_extract = False
#resize to:
size = (256,256)
#use pretrained or not
use_pretrained = True
classes = np.load(image_classes, allow_pickle=True).item()
num_classes = len(classes)
#train-test split
split = 0.85
#model save name
model_save_name = (model_name + "_" +
                   ("freeze" if feature_extract else "nofreeze") + "_" +
                   ("aug" if use_augmented else "noaug") + "_" +
                   f"seed{curr_seed}"
                   )



#Define some standard transformations
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((size))
    ])
## Probably better to follow the original resnet transformations
#See: (model.ResNet152_Weights.IMAGENET1K_V1.transforms)


full_dataset = CSVDataset(
    csv_file=str(dirpath / "train_images.csv"),
    base_dir=str(dirpath), transform = transformations
)
loader = DataLoader(full_dataset, batch_size=64, shuffle=True)

#check if it does what I want it to
# 1. Check dataset length
print(f"Dataset size: {len(full_dataset)}")

# 2. Get a single sample
image, label = full_dataset[0]
print(f"Single image shape: {image.shape}")  # Should be [3, 224, 224]
print(f"Single image type: {type(image)}")   # Should be torch.Tensor
print(f"Single label: {label}")              # Should be an integer
print(f"Label type: {type(label)}")          # Should be int or numpy.int64

# 3. Check a batch from the DataLoader
batch_images, batch_labels = next(iter(loader))
print(f"\nBatch images shape: {batch_images.shape}")  # Should be [32, 3, 224, 224]
print(f"Batch images type: {type(batch_images)}")     # Should be torch.Tensor
print(f"Batch images dtype: {batch_images.dtype}")    # Should be torch.float32
print(f"Batch labels shape: {batch_labels.shape}")    # Should be [32]
print(f"Batch labels type: {type(batch_labels)}")     # Should be torch.Tensor
print(f"Batch labels dtype: {batch_labels.dtype}")    # Could be torch.int64

# Initialize model

model_ft = None
# Initialize model
if model_name == "resnet152":
  """Resnet152"""
  ResNet_Weights = models.ResNet152_Weights.DEFAULT
  model_transforms = ResNet_Weights.transforms()
  if use_pretrained:
    model_ft = models.resnet152(weights=ResNet_Weights)
  else:
    model_ft = models.resnet152()
  num_ftrs = model_ft.fc.in_features
  model_ft.fc = nn.Linear(num_ftrs, num_classes)
if model_name == "resnet50":
  """Resnet50"""
  ResNet_Weights = models.ResNet50_Weights.DEFAULT
  model_transforms = ResNet_Weights.transforms()
  if use_pretrained:
    model_ft = models.resnet50(weights=ResNet_Weights)
  else:
    model_ft = models.resnet50()
  num_ftrs = model_ft.fc.in_features
  model_ft.fc = nn.Linear(num_ftrs, num_classes)
if model_name == "squeezenet":
  """Squeezenet"""
  squeezenet1_0_weights = models.SqueezeNet1_0_Weights.DEFAULT
  model_transforms = squeezenet1_0_weights.transforms()
  if use_pretrained:
    model_ft = models.squeezenet1_0(weights=squeezenet1_0_weights)
  else:
    model_ft = models.squeezenet1_0()
  set_parameter_requires_grad(model_ft, feature_extract)
  model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
  model_ft.num_classes = num_classes
if model_name == "efficientnetb0":
  """Efficientnetb0"""
  efficientnet_b0_weights = models.EfficientNet_B0_Weights.DEFAULT
  model_transforms = efficientnet_b0_weights.transforms()
  if use_pretrained:
    model_ft = models.efficientnet_b0(weights=efficientnet_b0_weights)
  else:
    model_ft = models.efficientnet_b0()
  set_parameter_requires_grad(model_ft, feature_extract)
  num_ftrs = model_ft.classifier[1].in_features
  model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
  model_ft.num_classes = num_classes
if model_name == "efficientnetV2s":
  """EfficientnetV2s"""
  efficientnet_v2_s_weights = models.EfficientNet_V2_S_Weights.DEFAULT
  model_transforms = efficientnet_v2_s_weights.transforms()
  if use_pretrained:
    model_ft = models.efficientnet_v2_s(weights=efficientnet_v2_s_weights)
  else:
    model_ft = models.efficientnet_v2_s()
  set_parameter_requires_grad(model_ft, feature_extract)
  num_ftrs = model_ft.classifier[1].in_features
  model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
  model_ft.num_classes = num_classes
if model_name == "SwinV2t":
  """SwinV2t"""
  swin_v2_t_weights = models.Swin_V2_T_Weights.DEFAULT
  model_transforms = swin_v2_t_weights.transforms()
  if use_pretrained:
    model_ft = models.swin_v2_t(weights=swin_v2_t_weights)
  else:
    model_ft = models.swin_v2_t()

if model_name == "ensem3avg":
    model_ft = ModelM3AVG()
if model_name == "ensem3max":
    model_ft = ModelM3MAX()
if model_name == "cnn":
    model_ft = CNN()

#set_parameter_requires_grad(model_ft, feature_extract)
#num_ftrs = model_ft.head.in_features
#model_ft.head = nn.Linear(num_ftrs, num_classes)
#model_ft.num_classes = num_classes


  # Train-validation split
# Split into train (85%) and validation (15%)
train_size = int(split * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(curr_seed)
)


# data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)



# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


#put model on device
model_ft = model_ft.to(device)


#gather optimizable parameters
params_to_update = model_ft.parameters()
#Design optimzer
optimizer = optim.SGD(model_ft.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()


# Train and evaluate
model_trained, hist = train_model(model_ft,
                            train_loader,
                            val_loader,
                            criterion,
                            optimizer,
                            schedular=None,
                            num_epochs=num_epochs,
                            device=device)


torch.save(model_trained.state_dict(), "model3max3.pth")