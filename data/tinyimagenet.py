import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import os, glob, random, numpy as np, time
from torchvision.io import read_image, ImageReadMode

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, id_dict, transform=None):
        self.filenames = sorted(glob.glob(os.path.join(root_dir, '*/*/*.JPEG')))
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

class ValTinyImageNetDataset(Dataset):
    def __init__(self, root_dir, id_dict, transform=None):
        self.filenames = sorted(glob.glob(os.path.join(root_dir, 'images/*.JPEG')))
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

def get_dataloaders(data_root, batch_size=128, num_workers=8, pin_memory=True, val_split=0.1, random_seed=None):
    # Train/val 분할은 항상 고정(seed=42)
    if random_seed is None:
        random_seed = int(time.time()) % (2**32)
    print(f"Using random seed: {random_seed}")  # 확인용 출력
    
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
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train_dataset = TinyImageNetDataset(
        os.path.join(data_root, 'train'),
        id_dict=id_dict,
        transform=train_transform
    )

    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, valid_idx)

    test_dataset = ValTinyImageNetDataset(
        os.path.join(data_root, 'val'),
        id_dict=id_dict,
        transform=valid_transform
    )

    # 매번 다른 랜덤 시드를 설정하여 학습 randomness를 추가
    random_seed = int(time.time())
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    print("Train dataset size:", len(train_idx))
    print("Validation dataset size:", len(valid_idx))
    print("Test dataset size:", len(test_dataset))

    return train_loader, val_loader, test_loader
