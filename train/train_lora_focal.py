import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import time
from datetime import timedelta
import sys
sys.path.append("/SSDe/youmin_park/adapter-weight-ensemble/")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from util import read_conf, validation_accuracy, validation_accuracy_lora #, calculate_flops
import random
import rein

import dino_variant
from sklearn.metrics import f1_score
from data import cifar10, cifar100, ham10000
from losses import RankMixup_MNDCG, RankMixup_MRL, focal_loss, focal_loss_adaptive_gamma

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_requires_grad(model, layers_to_train):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_train):
            param.requires_grad = True
        else:
            param.requires_grad = False
            
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cifar100')
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    # parser.add_argument('--save_path', '-s', type=str)
    # parser.add_argument('--noise_rate', '-n', type=float, default=0.2)
    args = parser.parse_args()

    # config = utils.read_conf('conf/'+args.data+'.json')
    config = read_conf('conf/data/'+args.data+'.yaml')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    # max_epoch = int(config['epoch'])
    max_epoch = 100
    # noise_rate = args.noise_rate

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch), int(0.9*max_epoch)]


    if args.data == 'cifar10':
        train_loader, valid_loader = cifar10.get_train_valid_loader(batch_size, augment=True, random_seed=42, valid_size=0.1, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
    elif args.data == 'cifar100':
        train_loader, valid_loader = cifar100.get_train_valid_loader(data_dir=data_path, augment=True, batch_size=32, valid_size=0.1, random_seed=42, shuffle=True, num_workers=4, pin_memory=True)
    elif args.data == 'ham10000':
        train_loader, valid_loader, _ = ham10000.get_dataloaders(data_path, batch_size=32, num_workers=4)
    # elif args.data == 'bloodmnist':
    #     train_loader, valid_loader, _ = bloodmnist.get_dataloader(batch_size=32, download=True, num_workers=4)
    # elif args.data == 'pathmnist':
    #     train_loader, valid_loader, _ = pathmnist.get_dataloader(batch_size=32, download=True, num_workers=4)
    # elif args.data == 'retinamnist':
    #     train_loader, valid_loader, _ = retinamnist.get_dataloader(batch_size=32, download=True, num_workers=4)
    
        
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    elif args.netsize == 'l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant




    dino = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = dino.state_dict()
    new_state_dict = dict()

    for k in dino_state_dict.keys():
        new_k = k.replace("attn.qkv", "attn.qkv.qkv")
        new_state_dict[new_k] = dino_state_dict[k]
        
    model = rein.LoRADinoVisionTransformer(dino)
    model.dino.load_state_dict(new_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)
    
    print(model)
    
    # num_params = count_trainable_params(model)
    # print(f"Number of trainable parameters: {num_params}")
    
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = focal_loss.FocalLoss(gamma=3) #gamma 커지면 easy sample에 대한 loss 감소
    # criterion = focal_loss_adaptive_gamma.FocalLossAdaptive()
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-6)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 1) 

    model.train()
    print('model: ', sum(p.numel() for p in model.parameters()))
    print('model_adapter: ', sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'linear' in n))
    
    scaler = GradScaler()
    avg_accuracy = 0.0
    start_time = time.time()
    
    for epoch in range(max_epoch):
        epoch_start_time = time.time()
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if targets.ndim > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)
                
            if targets.ndim > 1:
                targets = targets.view(-1) 
            optimizer.zero_grad()
            
            with autocast(enabled=True):
                features = model.forward_features(inputs)
                outputs = model.linear(features)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()             
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')
            
            train_accuracy = correct/total
                  
        train_avg_loss = total_loss/len(train_loader)
        epoch_duration = time.time() - epoch_start_time
        epoch_time = str(timedelta(seconds=epoch_duration))
        remaining_time = (max_epoch - (epoch + 1)) * epoch_duration
        formatted_remaining_time = str(timedelta(seconds=remaining_time))
        print(f"\nEpoch {epoch} took {epoch_time}")
        print(f"Estimated remaining training time: {formatted_remaining_time}")
        print()

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0

        valid_accuracy = validation_accuracy_lora(model, valid_loader, device)
        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy 
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())
    
    total_duration = time.time() - start_time
    totoal_time = str(timedelta(seconds=total_duration))
    print(f"Total training time: {totoal_time}")

    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))
    
if __name__ =='__main__':
    train()