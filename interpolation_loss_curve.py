import os
import sys
sys.path.append("/SSDe/youmin_park/adapter-weight-ensemble/")
import glob
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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

def safe_forward(model, inputs, args):
    model.eval()
    with torch.no_grad():
        outputs = model.forward_features(inputs)
        
        # Handle if output is dict
        if isinstance(outputs, dict):
            if 'x_norm_clstoken' in outputs:
                outputs = outputs['x_norm_clstoken']
            elif 'cls' in outputs:
                outputs = outputs['cls']
            else:
                raise ValueError(f"Unknown dict structure in forward_features: {outputs.keys()}")

        # Handle by adapter type
        if args.adapter == 'rein':
            outputs = model.linear(outputs[:, 0, :])
        elif args.adapter == 'lora':
            outputs = model.linear(outputs[:, 0, :])
        elif args.adapter == 'adaptformer':
            outputs = model.linear(outputs[:, 0, :])
        else:
            raise ValueError(f"Unknown adapter type: {args.adapter}")

        outputs = torch.softmax(outputs, dim=1)
        
    return outputs

def rein_forward(model, inputs):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    # output = torch.softmax(output, dim=1)
    return output

def lora_forward(model, inputs):
    with autocast(enabled=True):
        features = model.forward_features(inputs)
        output = model.linear(features)
        # output = torch.softmax(output, dim=1)
    return output

def adaptformer_forward(model, inputs):
    f = model.forward_features(inputs)[:, 0, :]
    output = model.linear(f)
    # outputs = torch.softmax(outputs, dim=1) 
    return output

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
# parser.add_argument('--branch', default='yes', type=str)
# parser.add_argument('--save_path', '-s', type=str)
args = parser.parse_args()

config = read_conf('conf/data/'+args.data+'.yaml')
device = torch.device(f"cuda:{args.gpu}")
# torch.cuda.set_device(device)  

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
    
def load_model(dino_state_dict, variant, args):
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
        
    return model
    
# model.cuda()

branch_models = []
indep_models = []

for i in range(2):
    load_model(dino_state_dict, variant, args)
    save_path = os.path.join(config['save_path'], f'{args.adapter}_branch')
    ckpt_list = sorted(glob.glob(os.path.join(save_path, "cyclic_checkpoint_epoch*.pth")))
    state_dict = torch.load(ckpt_list[0], map_location='cpu')
    model.load_state_dict(state_dict, strict=False) 
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.cuda()  

    branch_models.append(model)    
    
for i in range(2):
    load_model(dino_state_dict, variant, args)
    save_path = os.path.join(config['save_path'], f'{args.adapter}_focal_{i+1}')
    state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=False) 
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.cuda()  

    indep_models.append(model)
    
print("Branch Models:", len(branch_models))
print("Independent Models:", len(indep_models))
    



def interpolate_models(model1, model2, alpha):
    """Linear interpolate between two models."""
    new_model = copy.deepcopy(model1)  # model1을 deep copy

    with torch.no_grad():
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            interpolated = (1 - alpha) * param1.data + alpha * param2.data
            new_model.state_dict()[name1].copy_(interpolated)

    return new_model


def compute_loss(model, dataloader, device):
    model.eval()
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = focal_loss.FocalLoss(gamma=3.0)
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = safe_forward(model, inputs, args)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    return total_loss / total_samples


def plot_interpolation_loss_curves(branch_models, indep_models, dataloader, device, steps=20):
    """Compare interpolation loss curves between branch soup models and independent models."""
    alphas = np.linspace(0, 1, steps)

    branch_losses = []
    indep_losses = []

    branch_model1, branch_model2 = branch_models
    indep_model1, indep_model2 = indep_models

    print("Computing Branch Soup Interpolation...")
    for alpha in tqdm(alphas):
        interpolated_model = interpolate_models(branch_model1, branch_model2, alpha).to(device)
        loss = compute_loss(interpolated_model, dataloader, device)
        branch_losses.append(loss)

    print("Computing Independent Models Interpolation...")
    for alpha in tqdm(alphas):
        interpolated_model = interpolate_models(indep_model1, indep_model2, alpha).to(device)
        loss = compute_loss(interpolated_model, dataloader, device)
        indep_losses.append(loss)

    plt.figure(figsize=(10, 7))
    plt.plot(alphas, branch_losses, label="Branch Soup Interpolation", marker='o')
    plt.plot(alphas, indep_losses, label="Independent Models Interpolation", marker='x')
    plt.xlabel("Interpolation Coefficient (alpha)")
    plt.ylabel("Validation Loss")
    plt.title("Interpolation Loss Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("interpolation_loss_curve_comparison.png")
    plt.show()




plot_interpolation_loss_curves(branch_models, indep_models, test_loader, device, steps=20)

