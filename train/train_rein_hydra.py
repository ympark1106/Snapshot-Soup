import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import sys
sys.path.append("/SSDe/youmin_park/adapter-weight-ensemble/")
import time
from datetime import timedelta

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
from utils import read_conf, validation_accuracy

import random
import rein

import dino_variant
from sklearn.metrics import f1_score
from data import cifar100, ham10000, eyepacs
from losses import RankMixup_MNDCG, RankMixup_MRL, focal_loss, focal_loss_adaptive_gamma

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
    args = parser.parse_args()
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = read_conf('conf/data/'+args.data+'.yaml')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    num_workers = int(config['num_workers'])
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # if args.data == 'cifar10':
    #     train_loader, valid_loader = cifar10.get_train_valid_loader(batch_size, augment=True, random_seed=42, valid_size=0.1, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
    if args.data == 'cifar100':
        train_loader, valid_loader = cifar100.get_train_valid_loader(data_dir=data_path, augment=True, batch_size=batch_size, valid_size=0.1, random_seed=42, shuffle=True, num_workers=4, pin_memory=True)
    # elif args.data == 'cub':
    #     train_loader, valid_loader = cub.get_train_val_loader(data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=8, pin_memory=True)
    elif args.data == 'ham10000':
        train_loader, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=batch_size, num_workers=4)
    elif args.data == 'eyepacs':
        train_loader, valid_loader, test_loader = eyepacs.get_dataloaders(data_path, batch_size=batch_size, pin_memory=True,num_workers=num_workers)
    # elif args.data == 'bloodmnist':
    #     train_loader, valid_loader,_ = bloodmnist.get_dataloader(batch_size, download=True, num_workers=4)
    # elif args.data == 'pathmnist':
    #     train_loader, valid_loader,_ = pathmnist.get_dataloader(batch_size, download=True, num_workers=4)
    # elif args.data == 'retinamnist':
    #     train_loader, valid_loader,_ = retinamnist.get_dataloader(batch_size, download=True, num_workers=4)
        
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    elif args.netsize == 'l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant
    
    model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model.state_dict()

    model = rein.ReinsDinoVisionTransformer(
        **variant
    )
    set_requires_grad(model, ["reins", "linear"])
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)
    
    print(model)
    
    print("Max epoch: ", max_epoch)
    
    criterion = focal_loss.FocalLoss(gamma=3) 
    # criterion = torch.nn.CrossEntropyLoss()
    print("Criterion: ", criterion)
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)
    
    lr_decay_epochs = 70
    lr_scheduler_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)

    cyclic_start_epoch = lr_decay_epochs  
    cycle_length = 30        
    cyclic_epochs = max_epoch - cyclic_start_epoch  
    print(f"Total cyclic epochs: {cyclic_epochs}")

    checkpoint_path = os.path.join(save_path, f'checkpoint_epoch_{cyclic_start_epoch}.pth')  
    cyclic_scheduler = None  
    
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 1) 

    if not os.path.exists(checkpoint_path):
        print(f"Saving checkpoint for epoch {cyclic_start_epoch}")
        torch.save(model.state_dict(), checkpoint_path)

    avg_accuracy = 0.0
    start_time = time.time()

    for epoch in range(max_epoch):
            
        # 싸이클마다 70번째 에포크 상태로 되돌아감
        if epoch >= cyclic_start_epoch and (epoch - cyclic_start_epoch) % cycle_length == 0:
            print(f"\nRestoring model to checkpoint from epoch {cyclic_start_epoch}")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # DataParallel 모델에서 저장된 경우, 키에서 "module." 제거
            new_state_dict = {}
            for k, v in checkpoint.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v

            model.load_state_dict(new_state_dict, strict=False)  # strict=False 설정

            cyclic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=cycle_length, T_mult=1, eta_min=1e-5
            )

        epoch_start_time = time.time()
        ## Training
        model.train()
        total_loss = 0
        total = 0
        correct = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            targets = targets.type(torch.LongTensor)
            inputs, targets = inputs.to(device), targets.to(device)
            
            if targets.ndim > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)
            if targets.ndim > 1:
                targets = targets.view(-1)
            
            optimizer.zero_grad()

            features = model.forward_features(inputs)
            features = features[:, 0, :]
            outputs = model.linear(features)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)        
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')
            train_accuracy = correct/total
                
        if epoch < lr_decay_epochs:
            lr_scheduler_decay.step()
        else:
            # Cyclical LR에서 학습률이 저점(base_lr)에 도달했을 때 가중치 저장
            if optimizer.param_groups[0]['lr'] <= 0.00002:
                            torch.save(model.state_dict(), os.path.join(save_path, f'cyclic_checkpoint_epoch{epoch}.pth'))
            cyclic_scheduler.step()
            
        train_avg_loss = total_loss/len(train_loader)
        epoch_duration = time.time() - epoch_start_time
        epoch_time = str(timedelta(seconds=epoch_duration))
        remaining_time = (max_epoch - (epoch + 1)) * epoch_duration
        formatted_remaining_time = str(timedelta(seconds=remaining_time))
        print(f"\nEpoch {epoch} took {epoch_time}")
        print(f"Estimated remaining training time: {formatted_remaining_time}")
        print()
        print()

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0

        valid_accuracy = validation_accuracy(model, valid_loader, device)
        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy 

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        
        print(f'Epoch {epoch + 1}/{max_epoch} | Loss: {train_avg_loss:.4f} | '
            f'Train Acc: {train_accuracy:.4f} | Valid Acc: {valid_accuracy:.4f} | '
            f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
    total_duration = time.time() - start_time
    totoal_time = str(timedelta(seconds=total_duration))
    print(f"Total training time: {totoal_time}")

    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))
    
if __name__ =='__main__':
    train()