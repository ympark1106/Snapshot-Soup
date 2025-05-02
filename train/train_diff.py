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
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import functional as F

import argparse
import timm
import numpy as np
from util import read_conf, validation_accuracy, validation_accuracy_lora

import random
import rein
import adaptformer

import dino_variant
from sklearn.metrics import f1_score
from data import dataloader
from losses import RankMixup_MNDCG, RankMixup_MRL, focal_loss, focal_loss_adaptive_gamma

def rein_forward(model, inputs):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    output = torch.softmax(output, dim=1)
    return output

def lora_forward(model, inputs):
    with autocast(enabled=True):
        features = model.forward_features(inputs)
        output = model.linear(features)
        output = torch.softmax(output, dim=1)
    return output

def adaptformer_forward(model, inputs):
    f = model.forward_features(inputs)
    outputs = model.linear(f)
    outputs = torch.softmax(outputs, dim=1) 
    return outputs


def seed_for_init(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def set_requires_grad(model, layers_to_train):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_train):
            param.requires_grad = True
        else:
            param.requires_grad = False
            
def train():
    seed_for_init(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cifar100')
    parser.add_argument('--adapter', '-a', type=str, default='rein')
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate') 
    
    args = parser.parse_args()
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = read_conf('conf/data/'+args.data+'.yaml')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    max_epoch = 100
    # num_workers = int(config['num_workers'])
    
    train_loader, valid_loader, test_loader = dataloader.setup_data_loaders(args, data_path, batch_size)    
    
    os.makedirs(save_path, exist_ok=True)
        
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
        
    if args.adapter == 'rein':
        model = rein.ReinsDinoVisionTransformer(
            **variant
        )
        model.load_state_dict(dino_state_dict, strict=False)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    elif args.adapter == 'lora':
        new_state_dict = dict()

        for k in dino_state_dict.keys():
            new_k = k.replace("attn.qkv", "attn.qkv.qkv")
            new_state_dict[new_k] = dino_state_dict[k]
            
        model = rein.LoRADinoVisionTransformer(model)
        model.dino.load_state_dict(new_state_dict, strict=False)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    elif args.adapter == 'adaptformer':
        tuning_config = argparse.Namespace()
        # Adaptformer
        tuning_config.ffn_adapt = True
        tuning_config.ffn_num = 64
        tuning_config.ffn_option="parallel"
        tuning_config.ffn_adapter_layernorm_option="none"
        tuning_config.ffn_adapter_init_option="lora"
        tuning_config.ffn_adapter_scalar="0.1"
        tuning_config.d_model=384 # base -> 768
        # VPT
        tuning_config.vpt_on = False
        tuning_config.vpt_num = 1

        tuning_config.fulltune = False
            
        new_state_dict = dict()
        for k in dino_state_dict.keys():
            new_k = k.replace("mlp.", "")
            new_state_dict[new_k] = dino_state_dict[k]
        extra_tokens = dino_state_dict['pos_embed'][:, :1]
        src_weight = dino_state_dict['pos_embed'][:, 1:]
        src_weight = src_weight.reshape(1, 37, 37, 384).permute(0, 3, 1, 2)
        # src_weight = src_weight.reshape(1, 37, 37, 768).permute(0, 3, 1, 2) ＃ for base model

        dst_weight = F.interpolate(
            src_weight.float(), size=16, align_corners=False, mode='bilinear') # base model -> 16
        dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
        dst_weight = dst_weight.to(src_weight.dtype)
        new_state_dict['pos_embed'] = torch.cat((extra_tokens, dst_weight), dim=1)
        model = adaptformer.VisionTransformer(patch_size=14, embed_dim= 384, tuning_config = tuning_config, use_dinov2=True)
        model.load_state_dict(new_state_dict, strict=False) 
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    
    model.to(device)
    
    if args.adapter == 'adaptformer' or args.adapter == 'lora':
        set_requires_grad(model, ['adapt', 'linear', 'embeddings'])
    elif args.adapter == 'rein':
        set_requires_grad(model, ['reins', 'linear'])
    
    
    print(model)
    
    print("Max epoch: ", max_epoch)
    
    criterion = focal_loss.FocalLoss(gamma=3) 
    # criterion = torch.nn.CrossEntropyLoss()
    print("Criterion: ", criterion)
    model.eval()


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
 
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 1) 

    if args.adapter == 'lora':
        scalar = GradScaler()
        
    avg_accuracy = 0.0
    start_time = time.time()
        
    for epoch in range(max_epoch):
        epoch_start_time = time.time()
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)           
            
            if targets.ndim > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)
                
            if targets.ndim > 1:
                targets = targets.view(-1) 
            
            optimizer.zero_grad()
            
            if args.adapter == 'rein':
                outputs = rein_forward(model, inputs)
            elif args.adapter == 'lora':
                with autocast(enabled=True):
                    outputs = lora_forward(model, inputs)
            elif args.adapter == 'adaptformer':
                outputs = adaptformer_forward(model, inputs)
                
            loss = criterion(outputs, targets)
            loss.backward()            
            optimizer.step()

            total_loss += loss
            total += targets.size(0)
            
            # _, predicted = outputs[:len(targets)].max(1)        
            _, predicted = outputs.max(1)    
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

        if args.adapter == 'rein' or args.adapter == 'adaptformer':
            valid_accuracy = validation_accuracy(model, valid_loader, device, mode=args.adapter)
        elif args.adapter == 'lora':
            valid_accuracy = validation_accuracy_lora(model, valid_loader, device)
            
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