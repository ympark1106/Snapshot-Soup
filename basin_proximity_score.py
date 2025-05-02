import os
import sys
sys.path.append("/SSDe/youmin_park/adapter-weight-ensemble/")
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
parser.add_argument('--branch', default='yes', type=str)
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

models = []


if args.branch == 'yes':
    for i in range(10):
        model = load_model(dino_state_dict, variant, args)
        save_path = os.path.join(config['save_path'], f'{args.adapter}_branch')
        ckpt_list = sorted(glob.glob(os.path.join(save_path, "cyclic_checkpoint_epoch*.pth")))
        state_dict = torch.load(ckpt_list[0], map_location='cpu')
        model.load_state_dict(state_dict, strict=False) 
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.cuda()  

        models.append(model)    
        
elif args.branch == 'no':
    for i in range(10):
        model = load_model(dino_state_dict, variant, args)
        save_path = os.path.join(config['save_path'], f'{args.adapter}_focal_{i+1}')
        state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict']
        model.load_state_dict(state_dict, strict=False) 
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.cuda()  

        models.append(model)
    

    
def compute_param_distance(model1, model2):
    distance = 0.0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        distance += torch.norm(p1.data - p2.data, p=2).item()
    return distance


def compute_loss(model, dataloader):
    model.eval()
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = focal_loss.FocalLoss(gamma=3.0)
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = forward(model, inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples


def compute_basin_proximity(model1, model2, loss1, loss2, lambda_param=1.0, lambda_loss=0):
    param_dist = compute_param_distance(model1, model2)
    loss_diff = abs(loss1 - loss2)
    return lambda_param * param_dist + lambda_loss * loss_diff


def compute_pairwise_basin_scores(models, losses, lambda_param=1.0, lambda_loss=1.0):
    n = len(models)
    score_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                score = compute_basin_proximity(models[i], models[j], losses[i], losses[j], lambda_param, lambda_loss)
                score_matrix[i, j] = score
    return score_matrix


def plot_heatmap(score_matrix, args, title="Basin Proximity Heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap="viridis")
    plt.title(title)
    plt.xlabel("Model Index")
    plt.ylabel("Model Index")
    if args.branch == 'yes':
        plt.savefig(os.path.join("vis_score",f"Branch_{args.data}__{args.adapter}_branch_basin_proximity_heatmap.png"))
    elif args.branch == 'no':
        plt.savefig(os.path.join("vis_score", f"{args.data}__{args.adapter}_basin_proximity_heatmap.png"))
    plt.show()



_, _, test_loader = dataloader.setup_data_loaders(args, data_path, batch_size)

losses = [compute_loss(model, test_loader) for model in tqdm(models)]

score_matrix = compute_pairwise_basin_scores(models, losses, lambda_param=1.0, lambda_loss=1.0)

for idx, loss_val in enumerate(losses):
    print(f"\nModel {idx}: Loss = {loss_val:.4f}")

print("\nPairwise Parameter Distances:")
for i in range(len(models)):
    for j in range(i+1, len(models)):
        param_dist = compute_param_distance(models[i], models[j])
        print(f"Distance between Model {i} and Model {j}: {param_dist:.4f}")

print("\nPairwise Basin Proximity Scores:")
for i in range(len(models)):
    for j in range(i+1, len(models)):
        score = compute_basin_proximity(models[i], models[j], losses[i], losses[j], lambda_param=1.0, lambda_loss=1.0)
        print(f"Basin Proximity between Model {i} and Model {j}: {score:.4f}")


plot_heatmap(score_matrix, args)


