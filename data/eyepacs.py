import os
import pandas as pd
import random
import torch
import time
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

# Custom dataset class
class EyePACS_Dataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0] + '.jpeg')
        image = Image.open(img_name).convert("RGB")
        label = self.data.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# APTOS2019 Dataset
class APTOS2019_Dataset(Dataset):
    def __init__(self, csv_files, root_dirs, transform=None):
        dataframes = [pd.read_csv(csv) for csv in csv_files]
        self.data = pd.concat(dataframes, ignore_index=True)
        self.root_dirs = root_dirs
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = None
        for root_dir in self.root_dirs:
            img_path = os.path.join(root_dir, self.data.iloc[idx, 0] + '.png')
            if os.path.exists(img_path):
                img_name = img_path
                break
        
        if img_name is None:
            raise FileNotFoundError(f"Image {self.data.iloc[idx, 0]} not found in given directories")
        
        image = Image.open(img_name).convert("RGB")
        label = self.data.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# DataLoader function
def get_dataloaders(data_dir, batch_size=32, num_workers=4, pin_memory=False, val_split=0.2, random_seed=None):
    if random_seed is None:
        random_seed = int(time.time()) % (2**32)
    print(f"Using random seed: {random_seed}")  # 확인용 출력

    # Set random seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # Image transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # File paths
    eyepacs_dir = os.path.join(data_dir, 'eyepacs')
    aptos_dir = os.path.join(data_dir, 'aptos2019')
    
    train_csv = os.path.join(eyepacs_dir, 'trainLabels.csv')
    
    aptos_csvs = [
        os.path.join(aptos_dir, 'train_1.csv'),
        os.path.join(aptos_dir, 'valid.csv'),
        os.path.join(aptos_dir, 'test.csv')
    ]
    
    train_dir = os.path.join(eyepacs_dir, 'train')
    aptos_dirs = [
        os.path.join(aptos_dir, 'train_images/train_images'),
        os.path.join(aptos_dir, 'val_images/val_images'),
        os.path.join(aptos_dir, 'test_images/test_images')
    ]
    
    # Load EyePACS data and split into train/val
    eyepacs_data = pd.read_csv(train_csv)
    train_data, val_data = train_test_split(eyepacs_data, test_size=val_split, random_state=random_seed, stratify=eyepacs_data.iloc[:, 1])
    
    # Datasets
    train_dataset = EyePACS_Dataset(data=train_data, root_dir=train_dir, transform=train_transform)
    val_dataset = EyePACS_Dataset(data=val_data, root_dir=train_dir, transform=test_transform)
    test_dataset = APTOS2019_Dataset(csv_files=aptos_csvs, root_dirs=aptos_dirs, transform=test_transform)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)
    
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))
    
    return train_loader, val_loader, test_loader