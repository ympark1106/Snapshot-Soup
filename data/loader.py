import os
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import random
import numpy as np
import torch
import json
import albumentations
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from albumentations import CLAHE
import imageio
from random import *

def get_label_downext(file):
    label_a = 0
    label_b = 0
    label_c = 0
    extract_label = 0
    img_name = file
    img_name_num = img_name.split("/")[3].split(".")[0].split("_")[0]
    teeth_num = img_name.split("/")[3].split(".")[0].split("_")[1]
    with open("./crop_dataset/Annotations/{}.jpg.png.json".format(img_name_num)) as json_file:
        json_data = json.load(json_file)
        json_obj = json_data["objects"]
        img_label = []

        for anno_num in range(len(json_obj)):
            object_class_id = json_data["objects"][anno_num]["classId"]
            object_class_Title = json_data["objects"][anno_num]["classTitle"]
            object_class_tags = json_data["objects"][anno_num]["tags"]
            if object_class_tags == 0:
                continue

            else:
                for tag in range(len(object_class_tags)):

                    if object_class_Title == "#{}".format(teeth_num):
                        object_class_name = json_data["objects"][anno_num]["tags"][tag]["name"]
                        img_label.append(object_class_name)
                    else:
                        pass
    if 'P.A' in img_label:
        label_a = 0
        
    elif 'P.B' in img_label:
        label_a = 1

    elif 'P.C' in img_label:
        label_a = 2
        
    # P.1, P.2, P.3
    if 'P.I' in img_label:
        label_b = 0
    
    elif 'P.II' in img_label:
        label_b = 1
    
    elif 'P.III' in img_label:
        label_b = 2
        
    # L.근심(m), L.수직(v), L.수평(h), L.역위(i), L.원심(d), L.협설(t) 
    if 'L.근심' in img_label:
        label_c = 0

    elif 'L.수직' in img_label:
        label_c = 1
    
    elif 'L.수평' in img_label:
        label_c = 2

    elif 'L.역위' in img_label:
        label_c = 3

    elif 'L.원심' in img_label:
        label_c = 4
    
    elif 'L.협설' in img_label:
        label_c = 5

    level_1 = ['0/0/1']
    level_2 = ['0/0/4' '0/1/1', '1/0/1', '1/0/4']
    level_3 = ['0/0/2', '0/0/0', '0/0/5', '0/1/2', '0/0/0', '0/1/4']

    extract_level = str(label_a) + "/" + str(label_b) + "/" + str(label_c)
    if extract_level in level_1:
        extract_label = 0
    elif extract_level in level_2:
        extract_label = 1
    elif extract_level in level_3:
        extract_label = 2
    else:
        extract_label = 3
    return extract_label

def function_downext(root_dir):
    img_label_pairs = []
    for file in root_dir:
        img_label_pair = []
        img_label_pair.append(file)
        label = get_label_downext(file)
        img_label_pair.append(label)
        img_label_pairs.append(img_label_pair)

    return img_label_pairs

class DownDataLoader_ext(Dataset):
    
    def __init__(self, root_dir, set="train", he='CLAHE', transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.set = set
        self.he = he
        self.imgs = function_downext(self.root_dir)

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, index):
        global image
        img_name = self.imgs[index][0]
        label = self.imgs[index][1]
        img_data = img_name.split("/")[-1].split(".")[0]
        self.set = img_name.split("/")[-2].split("_")[2]

        ori_img = cv2.imread("/data_3/JSLEE/Wisdom_classification_v5/crop_dataset/PNGImages_crop/{}.png".format(img_data), cv2.IMREAD_COLOR)          #추가

        if self.set == 'test':
            mask_img = np.asarray(Image.open("/data_3/JSLEE/Wisdom_classification_v5/crop_dataset/Infer_SegMap3_crop/{}.png".format(img_data)))
        else : 
            mask_img = np.asarray(Image.open("/data_3/JSLEE/Wisdom_classification_v5/crop_dataset/SegMap_crop/{}.png".format(img_data)))

        name = img_name.split("/")[3].split(".")[0]

        if self.he == 'CLAHE':
            clahe = CLAHE(clip_limit=(4.0,4.0), tile_grid_size=(8,8), p=1.0)
            data = {"image": ori_img}
            he_ori_img = clahe(**data)
            he_ori_img = he_ori_img["image"]


        transform_totensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ])
        transform_totensor2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485,], std = [0.229,])
                ])

        augmented = self.transform(image = he_ori_img, mask = mask_img)
        #augmented = self.transform(image = he_ori_img)
        image = augmented["image"]
        mask = augmented["mask"]
        image = transform_totensor(Image.fromarray((image*255).astype(np.uint8)))
        mask = transform_totensor2(Image.fromarray((mask*255).astype(np.uint8)))
        image = torch.cat((image, mask),0)


        return img_data, image, label
##################################################################################################################################
def get_label_downcom(file):
    label = 0
    img_name = file
    img_name_num = img_name.split("/")[3].split(".")[0].split("_")[0]
    teeth_num = img_name.split("/")[3].split(".")[0].split("_")[1]
    with open("./crop_dataset/Annotations/{}.jpg.png.json".format(img_name_num)) as json_file:
        json_data = json.load(json_file)
        json_obj = json_data["objects"]
        img_label = []

        for anno_num in range(len(json_obj)):
            object_class_id = json_data["objects"][anno_num]["classId"]
            object_class_Title = json_data["objects"][anno_num]["classTitle"]
            object_class_tags = json_data["objects"][anno_num]["tags"]
            if object_class_tags == 0:
                continue

            else:
                for tag in range(len(object_class_tags)):

                    if object_class_Title == "#{}".format(teeth_num):
                        object_class_name = json_data["objects"][anno_num]["tags"][tag]["name"]
                        img_label.append(object_class_name)
                    else:
                        pass
    #N.1, N.2, N.3, darkening
    if 'N.1' in img_label:
        label = 0
    
    elif 'N.2' in img_label:
        label = 1
    
        if 'N.2' and 'Darkening' in img_label:
            label = 2
        else:
            pass

    elif 'N.3' in img_label:
        label = 3
    
        if 'N.3' and 'Darkening' in img_label:
            label = 4
        else:
            pass

    if label == 0:
        complication_label = 0
    elif label == 1:
        complication_label = 1
    elif label == 2:
        complication_label = 1
    elif label == 3:
        complication_label = 2
    elif label == 4:
        complication_label = 2

    return complication_label

def function_downcom(root_dir):
    img_label_pairs = []
    for file in root_dir:
        img_label_pair = []
        img_label_pair.append(file)
        label = get_label_downcom(file)
        img_label_pair.append(label)
        img_label_pairs.append(img_label_pair)

    return img_label_pairs

class DownDataLoader_com(Dataset):
    
    def __init__(self, root_dir, set="train", he='CLAHE', transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.set = set
        self.he = he
        self.imgs = function_downcom(self.root_dir)

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, index):
        global image
        img_name = self.imgs[index][0]
        label = self.imgs[index][1]
        img_data = img_name.split("/")[-1].split(".")[0]
        self.set = img_name.split("/")[-2].split("_")[2]
  
        ori_img = cv2.imread("/data_3/JSLEE/Wisdom_classification_v5/crop_dataset/PNGImages_crop/{}.png".format(img_data), cv2.IMREAD_COLOR)          #추가

        if self.set == 'test':
            mask_img = np.asarray(Image.open("/data_3/JSLEE/Wisdom_classification_v5/crop_dataset/Infer_SegMap3_crop/{}.png".format(img_data)))
        else : 
            mask_img = np.asarray(Image.open("/data_3/JSLEE/Wisdom_classification_v5/crop_dataset/SegMap_crop/{}.png".format(img_data)))

        name = img_name.split("/")[3].split(".")[0]

        if self.he == 'CLAHE':
            clahe = CLAHE(clip_limit=(4.0,4.0), tile_grid_size=(8,8), p=1.0)
            data = {"image": ori_img}
            he_ori_img = clahe(**data)
            he_ori_img = he_ori_img["image"]

        transform_totensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ])
        transform_totensor2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485,], std = [0.229,])
                ])


        augmented = self.transform(image = he_ori_img, mask = mask_img)
        #augmented = self.transform(image = he_ori_img)
        image = augmented["image"]
        mask = augmented["mask"]
        image = transform_totensor(Image.fromarray((image*255).astype(np.uint8)))
        mask = transform_totensor2(Image.fromarray((mask*255).astype(np.uint8)))
        image = torch.cat((image, mask),0)
        




        return img_data, image, label
#######################################################################################################################
def get_label_upext(file):
    label_a = 0
    label_b = 0
    extract_label = 0
    img_name = file
    img_name_num = img_name.split("/")[3].split(".")[0].split("_")[0]
    teeth_num = img_name.split("/")[3].split(".")[0].split("_")[1]
    with open("./crop_dataset/Annotations/{}.jpg.png.json".format(img_name_num)) as json_file:
        json_data = json.load(json_file)
        json_obj = json_data["objects"]
        img_label = []

        for anno_num in range(len(json_obj)):
            object_class_id = json_data["objects"][anno_num]["classId"]
            object_class_Title = json_data["objects"][anno_num]["classTitle"]
            object_class_tags = json_data["objects"][anno_num]["tags"]
            if object_class_tags == 0:
                continue

            else:
                for tag in range(len(object_class_tags)):

                    if object_class_Title == "#{}".format(teeth_num):
                        object_class_name = json_data["objects"][anno_num]["tags"][tag]["name"]
                        img_label.append(object_class_name)
                    else:
                        pass
    #U.A U.B U.C
    if 'U.A' in img_label:
        label_a = 0
    
    elif 'U.B' in img_label:
        label_a = 1
    
    elif 'U.C' in img_label:
        label_a = 2
    
    # U.근심(m), U.수직(v), U.수평(h), U.역위(i), U.원심(d), U.협설(t)

    if 'U.근심' in img_label:
        label_b = 0
    
    elif 'U.수직' in img_label:
        label_b = 1
    
    elif 'U.수평' in img_label:
        label_b = 2

    elif 'U.역위' in img_label:
        label_b = 3
    
    elif 'U.원심' in img_label:
        label_b = 4
    
    elif 'U.협설' in img_label:
        label_b = 5
    
    # 발치 난이도 단순발치, 단순매복, 복잡매복, 완전매복
    # 상악동천공 매우 낮음, 낮음, 높음
    level_1 = ['0/1', '0/4']
    level_2 = ['0/0', '0/5', '1/0', '1/1', '1/4']
    level_3 = ['0/2', '1/2']

    extract_level = str(label_a) + "/" + str(label_b)
    if extract_level in level_1:
        extract_label = 0
    elif extract_level in level_2:
        extract_label = 1
    elif extract_level in level_3:
        extract_label = 1
    else:
        extract_label = 2
    
    return extract_label

def function_upext(root_dir):
    img_label_pairs = []
    for file in root_dir:
        img_label_pair = []
        img_label_pair.append(file)
        label = get_label_upext(file)
        img_label_pair.append(label)
        img_label_pairs.append(img_label_pair)

    return img_label_pairs

class UpDataLoader_ext(Dataset):
    
    def __init__(self, root_dir, set="train", he='CLAHE', transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.set = set
        self.he = he
        self.imgs = function_upext(self.root_dir)

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, index):
        global image
        img_name = self.imgs[index][0]
        label = self.imgs[index][1]
        img_data = img_name.split("/")[-1].split(".")[0]
        self.set = img_name.split("/")[-2].split("_")[2]

        ori_img = cv2.imread("/data_3/JSLEE/Wisdom_classification_v5/crop_dataset/PNGImages_crop/{}.png".format(img_data), cv2.IMREAD_COLOR)          #추가

        if self.set == 'test':
            mask_img = np.asarray(Image.open("/data_3/JSLEE/Wisdom_classification_v5/crop_dataset/Infer_SegMap3_crop/{}.png".format(img_data)))
        else : 
            mask_img = np.asarray(Image.open("/data_3/JSLEE/Wisdom_classification_v5/crop_dataset/SegMap_crop/{}.png".format(img_data)))

        name = img_name.split("/")[3].split(".")[0]

        if self.he == 'CLAHE':
            clahe = CLAHE(clip_limit=(4.0,4.0), tile_grid_size=(8,8), p=1.0)
            data = {"image": ori_img}
            he_ori_img = clahe(**data)
            he_ori_img = he_ori_img["image"]

        transform_totensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ])
        transform_totensor2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485,], std = [0.229,])
                ])

        augmented = self.transform(image = he_ori_img, mask = mask_img)
        #augmented = self.transform(image = he_ori_img)
        image = augmented["image"]
        mask = augmented["mask"]
        image = transform_totensor(Image.fromarray((image*255).astype(np.uint8)))
        mask = transform_totensor2(Image.fromarray((mask*255).astype(np.uint8)))
        image = torch.cat((image, mask),0)



        return image, label

#######################################################################################################################
def get_label_upcom(file):
    label = 0
    img_name = file
    img_name_num = img_name.split("/")[3].split(".")[0].split("_")[0]
    teeth_num = img_name.split("/")[3].split(".")[0].split("_")[1]
    with open("./crop_dataset/Annotations/{}.jpg.png.json".format(img_name_num)) as json_file:
        json_data = json.load(json_file)
        json_obj = json_data["objects"]
        img_label = []

        for anno_num in range(len(json_obj)):
            object_class_id = json_data["objects"][anno_num]["classId"]
            object_class_Title = json_data["objects"][anno_num]["classTitle"]
            object_class_tags = json_data["objects"][anno_num]["tags"]
            if object_class_tags == 0:
                continue

            else:
                for tag in range(len(object_class_tags)):

                    if object_class_Title == "#{}".format(teeth_num):
                        object_class_name = json_data["objects"][anno_num]["tags"][tag]["name"]
                        img_label.append(object_class_name)
                    else:
                        pass
    # S.1, S.2, S.3
    if 'S.1' in img_label:
        label = 0

    elif 'S.2' in img_label:
        label = 1

    elif 'S.3' in img_label:
        label = 2

    else:
        pass

    if label == 0:
        complication_label = 0
    elif label == 1:
        complication_label = 1
    elif label == 2:
        complication_label = 2


    return complication_label

def function_upcom(root_dir):
    img_label_pairs = []
    for file in root_dir:
        img_label_pair = []
        img_label_pair.append(file)
        label = get_label_upcom(file)
        img_label_pair.append(label)
        img_label_pairs.append(img_label_pair)

    return img_label_pairs

class UpDataLoader_com(Dataset):
    
    def __init__(self, root_dir, set="train", he='CLAHE', transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.set = set
        self.he = he
        self.imgs = function_upcom(self.root_dir)

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, index):
        global image
        img_name = self.imgs[index][0]
       
        label = self.imgs[index][1]
        img_data = img_name.split("/")[-1].split(".")[0] 
        self.set = img_name.split("/")[-2].split("_")[2]

        ori_img = cv2.imread("/data_3/JSLEE/Wisdom_classification_v5/crop_dataset/PNGImages_crop/{}.png".format(img_data), cv2.IMREAD_COLOR)          #추가

        if self.set == 'test':
            mask_img = np.asarray(Image.open("/data_3/JSLEE/Wisdom_classification_v5/crop_dataset/Infer_SegMap3_crop/{}.png".format(img_data)))
        else : 
            mask_img = np.asarray(Image.open("/data_3/JSLEE/Wisdom_classification_v5/crop_dataset/SegMap_crop/{}.png".format(img_data)))

        name = img_name.split("/")[3].split(".")[0]

        if self.he == 'CLAHE':
            clahe = CLAHE(clip_limit=(4.0,4.0), tile_grid_size=(8,8), p=1.0)
            data = {"image": ori_img}
            he_ori_img = clahe(**data)
            he_ori_img = he_ori_img["image"]

        transform_totensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ])
        transform_totensor2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485,], std = [0.229,])
                ])

        augmented = self.transform(image = he_ori_img, mask = mask_img)
        #augmented = self.transform(image = he_ori_img)
        image = augmented["image"]
        mask = augmented["mask"]
        image = transform_totensor(Image.fromarray((image*255).astype(np.uint8)))
        mask = transform_totensor2(Image.fromarray((mask*255).astype(np.uint8)))
        image = torch.cat((image, mask),0)


        return image, label