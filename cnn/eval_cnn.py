import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
sys.path.append("/SSDe/youmin_park/adapter-weight-ensemble/")
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
from utils import read_conf, validation_accuracy, evaluate, compute_aurc, compute_auroc, compute_fpr95

from torchvision import models

import random
import rein

import dino_variant
from data import cifar10, cifar100, cub, ham10000, bloodmnist, pathmnist


def resnet_forward(model, inputs):
    output = model(inputs)
    output = torch.softmax(output, dim=1)
    return output



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='ham10000')
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--type', '-t', default= 'resnet', type=str)
    args = parser.parse_args()

    config = read_conf('conf/data/'+args.data+'.yaml')

    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    # batch_size = int(config['batch_size'])
    batch_size = 128


    if not os.path.exists(save_path):
        os.mkdir(save_path)


    if args.data == 'cifar10':
        test_loader = cifar10.get_test_loader(batch_size, shuffle=False, num_workers=4, pin_memory=True, data_dir=data_path)
    elif args.data == 'cifar100':
        test_loader = cifar100.get_test_loader(data_dir=data_path, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        _, valid_loader = cifar100.get_train_valid_loader(data_dir=data_path, augment=True, batch_size=batch_size, valid_size=0.1, random_seed=42, shuffle=True, num_workers=4, pin_memory=True)
    elif args.data == 'ham10000':
        train_loader, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=32, num_workers=4)


    if args.type == 'resnet':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, config['num_classes'])  
        model = model.to(device)
    elif args.type == 'vgg':
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, config['num_classes']) 
        model = model.to(device)
    elif args.type == 'densenet':
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, config['num_classes']) 
        model = model.to(device)    
        
    
    state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict']  
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
            
    # print(model)

    ## validation
    test_accuracy = validation_accuracy(model, test_loader, device, mode=args.type)
    print("\n🔹 Accuracy Metrics 🔹")
    print('test acc:', test_accuracy)

    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(test_loader):
            # print(f"Batch {batch_idx} targets:", target)
            inputs, target = inputs.to(device), target.to(device)
            if args.type == 'resnet':
                output = resnet_forward(model, inputs)
            elif args.type == 'vgg':
                output = model(inputs)
                output = torch.softmax(output, dim=1)
            elif args.type == 'densenet':
                output = model(inputs)
                output = torch.softmax(output, dim=1)
                # print(output.shape
                
            outputs.append(output.cpu())
            targets.append(target.cpu())
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy()
    targets = targets.astype(int)
    evaluate(outputs, targets, verbose=True)



if __name__ =='__main__':
    train()