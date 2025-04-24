import os
import sys
sys.path.append("/SSDe/youmin_park/adapter-weight-ensemble/")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import argparse
import timm
import numpy as np
from util import read_conf, validation_accuracy, evaluate, validation_accuracy_lora, compute_aurc, compute_auroc, compute_fpr95, ece, sce, ace, tace, rmsce, reliability_diagram
from torch.cuda.amp.autocast_mode import autocast

import random
import rein
import adaptformer

import dino_variant
from data import dataloader
from losses import RankMixup_MNDCG, RankMixup_MRL, focal_loss, focal_loss_adaptive_gamma


def rein_forward(model, inputs):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    # output = torch.softmax(output, dim=1)
    return output

def lora_forward(model, inputs):
    with autocast(enabled=True):
        features = model.forward_features(inputs)[:, 0, :]
        output = model.linear(features)
        # output = torch.softmax(output, dim=1)
    return output

def adaptformer_forward(model, inputs):
    f = model.forward_features(inputs)[:, 0, :]
    outputs = model.linear(f)
    # outputs = torch.softmax(outputs, dim=1) 
    return outputs

def forward(model, inputs):
    model.eval()
    with torch.no_grad():
        if args.adapter == 'rein':
            return rein_forward(model, inputs)
        elif args.adapter == 'lora':
            return lora_forward(model, inputs)
        elif args.adapter == 'adaptformer':
            return adaptformer_forward(model, inputs)
        else:
            raise ValueError("Unknown adapter")

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', type=str, default='cifar100')
parser.add_argument('--adapter', '-a', type=str, default='rein')
parser.add_argument('--gpu', '-g', default = '0', type=str)
parser.add_argument('--netsize', default='s', type=str)
# parser.add_argument('--save_path', '-s', type=str)
args = parser.parse_args()

config = read_conf('conf/data/'+args.data+'.yaml')
device = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(device)  

data_path = config['data_root']
batch_size = int(config['batch_size'])
max_epoch = 100
# num_workers = int(config['num_workers'])

train_loader, valid_loader, test_loader = dataloader.setup_data_loaders(args, data_path, batch_size) 

# loss_fn = focal_loss.FocalLoss(gamma=3.0)
loss_fn = nn.CrossEntropyLoss()
    
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
    
model.cuda()

# model_1 = model
# model_2 = model

# save_path_1 = os.path.join(config['save_path'], 'branch_soup')
# state_dict_1 = torch.load(os.path.join(save_path_1, f'{args.adapter}_branch_soup.pth'), map_location=device)
# model_1.load_state_dict(state_dict_1, strict=False) 
# model_1 = model_1

# save_path_2 = os.path.join(config['save_path'], f'{args.adapter}_focal_1')
# state_dict_2 = torch.load(os.path.join(save_path_2, 'last.pth.tar'), map_location=device)['state_dict']
# model_2.load_state_dict(state_dict_2, strict=False)  
# model_2 = model_2


model_1 = rein.ReinsDinoVisionTransformer(**variant)
save_path_1 = os.path.join(config['save_path'], 'branch_soup')
state_dict_1 = torch.load(os.path.join(save_path_1, f'{args.adapter}_branch_soup.pth'), map_location=device)
model_1.load_state_dict(state_dict_1, strict=False) 
model_1.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
model_1.cuda()

model_2 = rein.ReinsDinoVisionTransformer(**variant)
save_path_2 = os.path.join(config['save_path'], f'{args.adapter}_focal_1')
state_dict_2 = torch.load(os.path.join(save_path_2, 'last.pth.tar'), map_location=device)['state_dict']
model_2.load_state_dict(state_dict_2, strict=False)  
model_2.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
model_2.cuda()

model = rein.ReinsDinoVisionTransformer(**variant)
model.load_state_dict(dino_state_dict, strict=False)  
model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
model.cuda()

def get_flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def set_flat_params(model, flat_params):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[idx:idx+numel].reshape(p.shape))
        idx += numel

def compute_loss(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = forward(model, inputs)
            # print("outputs.shape:", outputs.shape)
            # print("targets.shape:", targets.shape)
            print("outputs:", outputs)
            print("targets:", targets)
            print("loss:", loss_fn(outputs, targets))
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def visualize_2d_loss_landscape(model, model1, model2, dataloader, loss_fn, grid_size=21, radius=1.0):
    theta_0 = get_flat_params(model)
    theta_1 = get_flat_params(model1)
    theta_2 = get_flat_params(model2)
    
    print("theta_0[:5]:", theta_0[:5])
    print("theta_1[:5]:", theta_1[:5])
    print("diff norm:", (theta_1 - theta_0).norm())

    u = theta_1 - theta_0
    v = theta_2 - theta_0

    u = u / u.norm()
    v = v - (u @ v) * u  # 직교화
    v = v / v.norm()

    X, Y = np.meshgrid(np.linspace(-radius, radius, grid_size), np.linspace(-radius, radius, grid_size))
    Z = np.zeros_like(X)

    base_model = deepcopy(model).cuda()

    for i in range(grid_size):
        for j in range(grid_size):
            direction = X[i, j] * u + Y[i, j] * v
            new_params = theta_0 + direction
            set_flat_params(base_model, new_params)
            Z[i, j] = compute_loss(base_model, dataloader, loss_fn, device)

    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar()
    plt.title("2D Loss Landscape")
    plt.xlabel("Direction u")
    plt.ylabel("Direction v")
    
    plt.savefig('loss_landscape.png', dpi=300)
    plt.show()

print("Loss at (0,0):", compute_loss(model, test_loader, loss_fn, device))
visualize_2d_loss_landscape(model, model_1, model_2, test_loader, loss_fn, grid_size=21, radius=1.0)