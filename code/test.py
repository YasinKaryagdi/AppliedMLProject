import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import pandas as pd
from CSVDataset import CSVDataset
from pathlib import Path
import numpy as np 
from models.modelM3AVG import ModelM3AVG
from models.modelM3MAX import ModelM3MAX
from code.models.random import BirdNet
    

def predict(model, test_loader, device="cuda"):
    model.eval()
    preds_list = []
    ids_list = []

    with torch.no_grad():
        for inputs, labels, img_ids in tqdm(test_loader):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            #Readd 1 to label to give right predictions
            preds_list.extend(preds.cpu().numpy()+1)
            ids_list.extend(img_ids.numpy() + 1)
    
    return ids_list, preds_list

cwd = Path.cwd()
gitpath = cwd
dirpath = gitpath / "aml-2025-feathers-in-focus"
train_images_csv = dirpath / "train_images.csv"
train_images_folder = dirpath / "train_images"
image_classes = dirpath / "class_names.npy"

#Defining model and training variables
#model
model_name = 'resnet'
#Epochs
num_epochs = 1
#feature extraction option
feature_extract = False
#resize to:
size = (256,256)
#use pretrained or not
use_pretrained = True
classes = np.load(image_classes, allow_pickle=True).item()
num_classes = len(classes)
#train-test split
split = 0.85

#Define some standard transformations
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((size)),
    transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
    ])


#Get Test set
test_dataset = CSVDataset(
    csv_file=str(dirpath / "test_images_path.csv"),
    base_dir=str(dirpath),
    transform = transformations,
    return_id=True
)
test_image_ids = test_dataset.df['id'].tolist()
#Create dataloader
test_loader = DataLoader(test_dataset, batch_size=80, shuffle=False)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 3. Check a batch from the DataLoader
# batch_images, batch_labels = next(iter(test_loader))
# print(f"\nBatch images shape: {batch_images.shape}")  # Should be [32, 3, 224, 224]
# print(f"Batch images type: {type(batch_images)}")     # Should be torch.Tensor
# print(f"Batch images dtype: {batch_images.dtype}")    # Should be torch.float32
# print(f"Batch labels shape: {batch_labels.shape}")    # Should be [32]
# print(f"Batch labels type: {type(batch_labels)}")     # Should be torch.Tensor
# print(f"Batch labels dtype: {batch_labels.dtype}")    # Could be torch.int64

#load model
finetuned_model = ModelM3MAX()
#num_ftrs = finetuned_model.fc.in_features
#finetuned_model.fc = nn.Linear(num_ftrs, num_classes)
finetuned_model.load_state_dict(torch.load("./model3max2.pth"))
finetuned_model.to(device)


#run model
test_ids, test_preds = predict(finetuned_model, test_loader, device=device)

#generate submission.csv
submission = pd.DataFrame({
    "id": test_ids,
    "label": test_preds
})

submission.to_csv("./model3max2.csv", index=False)