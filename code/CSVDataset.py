from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, csv_file, base_dir, transform=None, return_id=False):
        self.df = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.transform = transform
        self.return_id = return_id  # Useful for test set where no labels exist

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # extract fields
        img_id = idx
        relative_path = row['image_path'].lstrip('/')  # safe
        label = row['label'] - 1   # shift to 0-based indexing

        # build full path
        img_path = os.path.join(self.base_dir, relative_path)

        # load
        image = Image.open(img_path).convert('RGB')

        # transform
        if self.transform:
            image = self.transform(image)

        # optionally return id
        if self.return_id:
            return image, label, img_id

        return image, label