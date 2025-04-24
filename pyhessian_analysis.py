#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

from __future__ import print_function

import json
import os
import sys
sys.path.append("/SSDe/youmin_park/adapter-weight-ensemble/")

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from PyHessian.utils import *
from PyHessian.density_plot import get_esd_plot

from PyHessian.pyhessian import hessian

import rein
import adaptformer
import dino_variant
from data import dataloader
from losses import focal_loss
from util import read_conf




def set_requires_grad(model, layers_to_train):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_train):
            param.requires_grad = True
        else:
            param.requires_grad = False



parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument(
    '--mini-hessian-batch-size',
    type=int,
    default=200,
    help='input batch size for mini-hessian batch (default: 200)')
parser.add_argument('--hessian-batch-size',
                    type=int,
                    default=200,
                    help='input batch size for hessian (default: 200)')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    help='random seed (default: 1)')
parser.add_argument('--data', '-d', type=str, default='cifar100')
parser.add_argument('--adapter', '-a', type=str, default='rein')
parser.add_argument('--gpu', '-g', default = '0', type=str)
parser.add_argument('--netsize', default='s', type=str)
parser.add_argument('--save_path', '-s', type=str)
parser.add_argument('--resume',
                    type=str,
                    default='',
                    help='get the checkpoint')

args = parser.parse_args()

config = read_conf(os.path.join('conf', 'data', f'{args.data}.yaml'))
device = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(device)  
data_path = config['data_root']
batch_size = int(config['batch_size'])
save_path = os.path.join(config['save_path'], args.save_path)


for arg in vars(args):
    print(arg, getattr(args, arg))

# get dataset
train_loader, valid_loader, test_loader = dataloader.setup_data_loaders(args, data_path, batch_size)
# Get the hessian data
##############
assert (args.hessian_batch_size % args.mini_hessian_batch_size == 0)
assert (50000 % args.hessian_batch_size == 0)
batch_num = args.hessian_batch_size // args.mini_hessian_batch_size

if batch_num == 1:
    for inputs, labels in train_loader:
        hessian_dataloader = (inputs, labels)
        break
else:
    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(train_loader):
        hessian_dataloader.append((inputs, labels))
        if i == batch_num - 1:
            break

# get model
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
    
if args.adapter == 'adaptformer' or args.adapter == 'lora':
    set_requires_grad(model, ['adapt', 'linear', 'embeddings'])
elif args.adapter == 'rein':
    set_requires_grad(model, ['reins', 'linear'])
    

model.cuda()
# model = torch.nn.DataParallel(model, device_ids=[int(args.gpu)]) 
# model = model.cuda() 

# criterion = nn.CrossEntropyLoss()  # label loss
criterion = focal_loss.FocalLoss(gamma=3) 

###################
# Get model checkpoint, get saving folder
###################
# state_dict = torch.load(os.path.join(save_path, f'{args.adapter}_branch_soup.pth'), map_location=device)
state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'), map_location=device)['state_dict']
model.load_state_dict(state_dict, strict=False)

######################################################
# Begin the computation
######################################################

# turn model to eval mode
model.eval()
if batch_num == 1:
    hessian_comp = hessian(model,
                           criterion,
                           data=hessian_dataloader,
                           cuda=device)
else:
    hessian_comp = hessian(model,
                           criterion,
                           dataloader=hessian_dataloader,
                           cuda=device)

print(
    '********** finish data londing and begin Hessian computation **********')

top_eigenvalues, _ = hessian_comp.eigenvalues()
trace = hessian_comp.trace()
density_eigen, density_weight = hessian_comp.density()

print('\n***Top Eigenvalues: ', top_eigenvalues)
print('\n***Trace: ', np.mean(trace))

get_esd_plot(density_eigen, density_weight)
