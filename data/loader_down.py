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

from random import *

def get_label_downext(file):
    label_a = 0
    label_b = 0
    label_c = 0
    extract_label = 0
    img_name = file
    img_name_num = img_name.split("/")[-1].split(".")[0].split("_")[0]
    teeth_num = img_name.split("/")[-1].split(".")[0].split("_")[1]
  
    with open("./dataset/Annotations/{}.jpg.png.json".format(img_name_num)) as json_file:
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
    level_2 = ['0/0/4' '0/1/1', '1/0/1', '1/0/5']
    level_3 = ['0/0/2', '0/0/0', '0/0/5', '0/1/2', '0/1/0', '0/1/4']

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
    #return extract_label, img_label  ######
def function_downext(root_dir):
    img_label_pairs = []
    for file in root_dir:
        img_label_pair = []
        img_label_pair.append(file)
        label = get_label_downext(file)
        #label, label_ = get_label_downext(file) ####
        img_label_pair.append(label)
        #img_label_pair.append(label_)       #######
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
        #label_ = self.imgs[index][2]          #############

        img_data = img_name.split("/")[-1].split(".")[0]
        self.set = img_name.split("/")[-2].split("_")[0]

        ori_img = cv2.imread("/data2/JSLEE/Third_molar/classification/dataset/crop_PNG_Images/{}_man/{}.png".format(self.set, img_data), cv2.IMREAD_COLOR)          #추가

        if self.set == 'test':
            mask_img = np.asarray(Image.open("/data2/JSLEE/Third_molar/classification/dataset/crop_MASK_Images/{}_man_infer_0.005/{}.png".format(self.set, img_data)))
        else : 
            mask_img = np.asarray(Image.open("/data2/JSLEE/Third_molar/classification/dataset/crop_MASK_Images/{}_man/{}.png".format(self.set, img_data)))

        
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
##################################################################################################################################
def get_label_downcom(file):
    label = 0
    img_name = file
    img_name_num = img_name.split("/")[-1].split(".")[0].split("_")[0]
    teeth_num = img_name.split("/")[-1].split(".")[0].split("_")[1]
    with open("./dataset/Annotations/{}.jpg.png.json".format(img_name_num)) as json_file:
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
    else:
        label = 1
        
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
        self.set = img_name.split("/")[-2].split("_")[0]
        
        ori_img = cv2.imread("/data2/JSLEE/Third_molar/classification/dataset/crop_PNG_Images/{}_man/{}.png".format(self.set, img_data), cv2.IMREAD_COLOR)          #추가

        if self.set == 'test':
            mask_img = np.asarray(Image.open("/data2/JSLEE/Third_molar/classification/dataset/crop_MASK_Images/{}_man_infer_0.005/{}.png".format(self.set, img_data)))
        else : 
            mask_img = np.asarray(Image.open("/data2/JSLEE/Third_molar/classification/dataset/crop_MASK_Images/{}_man/{}.png".format(self.set, img_data)))

        name = img_name.split("/")[3].split(".")[0]

    
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
        image = transform_totensor(Image.fromarray((image*255).astype(np.uint8)))
        mask = augmented["mask"]
        mask = transform_totensor2(Image.fromarray((mask*255).astype(np.uint8)))
        image = torch.cat((image, mask),0)
        
        return image, label

