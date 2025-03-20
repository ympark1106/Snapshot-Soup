import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import os, glob
from torchvision.io import read_image, ImageReadMode

# Tiny ImageNet-200 Dataset Class
class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, id_dict, transform=None):
        self.filenames = glob.glob(os.path.join(root_dir, '*/*/*.JPEG'))
        self.transform = transform
        self.id_dict = id_dict

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)
        label = self.id_dict[img_path.split(os.sep)[-3]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

# Validation/Test Dataset Class using val folder
class ValTinyImageNetDataset(Dataset):
    def __init__(self, root_dir, id_dict, transform=None):
        self.filenames = glob.glob(os.path.join(root_dir, 'images/*.JPEG'))
        self.transform = transform
        self.id_dict = id_dict
        self.cls_dic = {}
        with open(os.path.join(root_dir, 'val_annotations.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img, cls_id = parts[0], parts[1]
                self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)
        label = self.cls_dic[os.path.basename(img_path)]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

# DataLoader Function
def get_dataloaders(data_root, batch_size=128, num_workers=8, pin_memory=True, random_seed=42, val_split=0.1):
    torch.manual_seed(random_seed)

    id_dict = {}
    with open(os.path.join(data_root, 'wnids.txt'), 'r') as f:
        for i, line in enumerate(f):
            id_dict[line.strip()] = i


    valid_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    full_train_dataset = TinyImageNetDataset(os.path.join(data_root, 'train'), id_dict=id_dict, transform=valid_transform)

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])


    test_dataset = ValTinyImageNetDataset(os.path.join(data_root, 'val'), id_dict=id_dict, transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))

    return train_loader, val_loader, test_loader