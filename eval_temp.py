import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os

import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
from util import read_conf, validation_accuracy, evaluate, validate, calculate_ece, calculate_nll, ece, sce, ace, tace, reliability_diagram

from util.temperature_scaling import ModelWithTemperature

import random
import rein
import torch.nn.functional as F
import dino_variant
from data import cifar10, cifar100, cub, ham10000, eyepacs, tinyimagenet


def rein_forward(model, inputs, temp=1.0, post_temp=False):
    if isinstance(model, ModelWithTemperature):
        logits = model.model.forward_features(inputs)[:, 0, :]
        logits = model.model.linear(logits)
    else:
        logits = model.forward_features(inputs)[:, 0, :]
        logits = model.linear(logits)

    if post_temp:
        if not isinstance(temp, torch.Tensor):
            temp = torch.tensor(temp, device=logits.device)
        temp = temp.to(logits.device)  # GPU로 이동
        logits = logits / temp
    
    return logits

def lora_forward(model, inputs, temp=1.0, post_temp=False):
    with torch.cuda.amp.autocast(enabled=True):
        features = model.forward_features(inputs)
        logits = model.linear(features)

    if post_temp:
        if not isinstance(temp, torch.Tensor):
            temp = torch.tensor(temp, device=logits.device)
        temp = temp.to(logits.device)  # GPU로 이동
        logits = logits / temp

    return logits

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cub')
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--type', '-t', default= 'rein', type=str)
    args = parser.parse_args()

    config = read_conf('conf/data/'+args.data+'.yaml')

    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    batch_size = int(config['batch_size'])


    if not os.path.exists(save_path):
        os.mkdir(save_path)


    if args.data == 'cifar10':
        test_loader = cifar10.get_test_loader(batch_size, shuffle=True, num_workers=4, pin_memory=True, data_dir=data_path)
        train_loader, valid_loader = cifar10.get_train_valid_loader(batch_size, augment=True, random_seed=42, valid_size=0.1, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
    elif args.data == 'cifar100':
        test_loader = cifar100.get_test_loader(data_dir=data_path, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        train_loader, valid_loader = cifar100.get_train_valid_loader(data_dir=data_path, augment=True, batch_size=batch_size, valid_size=0.1, random_seed=42, shuffle=True, num_workers=4, pin_memory=True)
    elif args.data == 'ham10000':
        train_loader, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=batch_size, num_workers=4)
    # elif args.data == 'bloodmnist':
    #     train_loader, test_loader, valid_loader = bloodmnist.get_dataloader(batch_size, download=True, num_workers=4)
    # elif args.data == 'pathmnist':
    #     train_loader, test_loader, valid_loader = pathmnist.get_dataloader(batch_size, download=True, num_workers=4)
    # elif args.data == 'retinamnist':
    #     train_loader, test_loader, valid_loader = retinamnist.get_dataloader(batch_size, download=True, num_workers=4)
    elif args.data == 'eyepacs':
        train_loader, valid_loader, test_loader = eyepacs.get_dataloaders(data_path, batch_size=batch_size, pin_memory=True,num_workers=16)
    elif args.data == 'tinyimagenet':
        train_loader, valid_loader, test_loader = tinyimagenet.get_dataloaders(data_path, batch_size=128, num_workers=4, pin_memory=True, val_split=0.1)
        
        
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant


    model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model.state_dict()

    if args.type == 'rein':
        model = rein.ReinsDinoVisionTransformer(
            **variant
        )

    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.load_state_dict(dino_state_dict, strict=False)
    model.to(device)

    # state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict']
    state_dict = torch.load(os.path.join(save_path, 'cyclic_checkpoint_epoch129.pth'), map_location=device)

    # state_dict = torch.load(os.path.join(save_path, f'Uniform_Soup_{args.data}.pth'), map_location=device)
    # state_dict = torch.load(os.path.join(save_path, f'Greedy_Soup_ACC_{args.data}.pth'), map_location=device)
    # state_dict = torch.load(os.path.join(save_path, f'Greedy_Soup_ECE_{args.data}.pth'), map_location=device)
    
    model.load_state_dict(state_dict, strict=False)
            
    model_temp = ModelWithTemperature(model)
    # print(model_temp)
    model_temp.set_temperature(valid_loader, cross_validate='ece', args=args)
    temp = model_temp.get_temperature()
    print(f"Optimal Temperature: {temp}")
    
    ## validation
    test_accuracy = validation_accuracy(model_temp, test_loader, device, mode=args.type)
    print('test acc:', test_accuracy)

    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(test_loader):
            # print(f"Batch {batch_idx} targets:", target)
            inputs, target = inputs.to(device), target.to(device)
            if args.type == 'rein':
                output = rein_forward(model_temp, inputs, temp=temp, post_temp=True)
                # print(output.shape)  
                probabilities = F.softmax(output, dim=1)  # Softmax로 확률 변환
            outputs.append(probabilities.cpu())
            targets.append(target.cpu())
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy()
    targets = targets.astype(int)
    evaluate(outputs, targets, verbose=True)

    ece_val = ece(targets, outputs, num_bins=15)
    sce_val = sce(targets, outputs, num_bins=15)
    ace_val = ace(targets, outputs, num_bins=15)
    tace_val = tace(targets, outputs, num_bins=15, threshold=0.01)
    
    print("\n🔹 Calibration Metrics (GCE 기반) 🔹")
    print(f"ECE  (Expected Calibration Error):           {ece_val * 100:.2f}%")
    print(f"SCE  (Static Calibration Error):             {sce_val * 100:.2f}%")
    print(f"ACE  (Adaptive Calibration Error):           {ace_val * 100:.2f}%")
    print(f"TACE (Thresholded Adaptive Calibration):     {tace_val * 100:.2f}%")

    reliability_diagram(outputs, targets, num_bins=15, title="ECE based", save_path="reliability_ece.png")
    


if __name__ =='__main__':
    train()