import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import timm
import numpy as np
from util import read_conf, validation_accuracy, evaluate, validation_accuracy_lora, compute_aurc, compute_auroc, compute_fpr95, ece, sce, ace, tace, rmsce, reliability_diagram
from torch.cuda.amp.autocast_mode import autocast

import random
import rein
import adaptformer

import dino_variant
from data import cifar10, cifar100, cub, ham10000, bloodmnist, pathmnist, retinamnist, eyepacs, tinyimagenet


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

def resnet_forward(model, inputs):
    output = model(inputs)
    output = torch.softmax(output, dim=1)
    return output



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cifar100')
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--net', '-n', default='dinov2', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--type', '-t', default= 'rein', type=str)
    args = parser.parse_args()

    config = read_conf('conf/data/'+args.data+'.yaml')

    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    # batch_size = 32
    # num_workers = int(config['num_workers'])

    if not os.path.exists(save_path):
        os.mkdir(save_path)


    if args.data == 'cifar100':
        test_loader = cifar100.get_test_loader(data_dir=data_path, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    elif args.data == 'ham10000':
        train_loader, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=batch_size, num_workers=4)
    elif args.data == 'eyepacs':
        train_loader, valid_loader, test_loader = eyepacs.get_dataloaders(data_path, batch_size=batch_size, pin_memory=True,num_workers=16)
    elif args.data == 'tinyimagenet':
        train_loader, valid_loader, test_loader = tinyimagenet.get_dataloaders(data_path, batch_size=128, num_workers=4, pin_memory=True, val_split=0.1)
                

    if args.net == 'dinov2':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant

        dino = torch.hub.load('facebookresearch/dinov2', model_load)
        dino_state_dict = dino.state_dict()


    elif args.net == 'dinov1':
        model_ = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        variant = dino_variant._dinov1_variant
        dino_state_dict = model_.state_dict()
        # print(dino_state_dict.keys())
        new_state_dict = dict()
        for k in dino_state_dict.keys():
            new_k = k.replace("mlp.", "")
            new_state_dict[new_k] = dino_state_dict[k]


    if args.type == 'linear':
        model = torch.hub.load('facebookresearch/dinov2', model_load)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.load_state_dict(dino_state_dict, strict=False)
        model.to(device)
    elif args.type == 'rein':
        model = rein.ReinsDinoVisionTransformer(
            **variant
        )
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.load_state_dict(dino_state_dict, strict=False)
        model.to(device)
    # elif args.type == 'rein_dropout':
    #     model = rein.ReinsDinoVisionTransformer_Dropout(
    #         **variant,
    #         dropout_rate=0.5
    #     )
    #     model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    #     model.load_state_dict(dino_state_dict, strict=False)
    #     model.to(device)
    elif args.type == 'lora':
        new_state_dict = dict()
        for k in dino_state_dict.keys():
            new_k = k.replace("attn.qkv", "attn.qkv.qkv")
            new_state_dict[new_k] = dino_state_dict[k]
        model = rein.LoRADinoVisionTransformer(dino)
        model.dino.load_state_dict(new_state_dict, strict=False)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)
    elif args.type == 'adaptformer':
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

    # print(model)

    # state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'), map_location=device)['state_dict']
    # state_dict = torch.load(os.path.join(save_path, f'{args.type}branch_soup.pth'), map_location=device)
    state_dict = torch.load(os.path.join(save_path, f'cyclic_checkpoint_epoch19.pth'), map_location=device)
    # state_dict = torch.load(os.path.join(save_path, 'checkpoint_epoch_70.pth'), map_location='cpu')
    
    # state_dict = torch.load(os.path.join(save_path, f'Uniform_Soup_{args.data}.pth'), map_location=device)
    # state_dict = torch.load(os.path.join(save_path, f'Greedy_Soup_ACC_{args.data}.pth'), map_location=device)
    # state_dict = torch.load(os.path.join(save_path, f'Greedy_Soup_ECE_{args.data}.pth'), map_location=device)
    
    model.load_state_dict(state_dict, strict=False)
    
    if args.type == 'rein_dropout':
        model.train() # MC Dropout
    else:
        model.eval()        
            
    # print(model)

    ## validation 
    if args.type == 'lora':
        test_accuracy = validation_accuracy_lora(model, test_loader, device)
    else:
        test_accuracy = validation_accuracy(model, test_loader, device, mode=args.type)
    print("\n🔹 Model Accuracy 🔹")
    print('test acc:', test_accuracy)

    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(test_loader):
            # print(f"Batch {batch_idx} targets:", target)
            inputs, target = inputs.to(device), target.to(device)
            if args.type == 'linear':
                output = model(inputs)
                output = model.linear(output)
                output = torch.softmax(output, dim=1)
            elif args.type == 'rein':
                output = rein_forward(model, inputs)
                # print(output.shape)
            elif args.type == 'resnet':
                output = resnet_forward(model, inputs)
            elif args.type == 'lora':
                with autocast(enabled=True):
                    output = lora_forward(model, inputs)
                    # print(output.shape)
            elif args.type == 'adaptformer':
                output = adaptformer_forward(model, inputs)
                # print(output.shape)
                
            outputs.append(output.cpu())
            targets.append(target.cpu())
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy()
    targets = targets.astype(int)
    evaluate(outputs, targets, verbose=True)
    
    
    # print("\n🔹 Calibration Metrics 🔹")

    ece_val = ece(targets, outputs, num_bins=15)
    sce_val = sce(targets, outputs, num_bins=15)
    ace_val = ace(targets, outputs, num_bins=15)
    tace_val = tace(targets, outputs, num_bins=15, threshold=0.01)
    
    print("\n🔹 Calibration Metrics (GCE 기반) 🔹")
    print(f"ECE  (Expected Calibration Error):           {ece_val * 100:.2f}%")
    print(f"SCE  (Static Calibration Error):             {sce_val * 100:.2f}%")
    print(f"ACE  (Adaptive Calibration Error):           {ace_val * 100:.2f}%")
    print(f"TACE (Thresholded Adaptive Calibration):     {tace_val * 100:.2f}%")

    # reliability_diagram(outputs, targets, num_bins=15, title="ECE based", save_path="reliability_ece.png")

    # Failure Prediction Metrics 계산
    # aurc = compute_aurc(outputs, targets)
    # auroc = compute_auroc(outputs, targets)
    # fpr95 = compute_fpr95(outputs, targets)
    
    # print("\n🔹 Failure Prediction Metrics 🔹")
    # print(f"AURC (Area Under Risk-Coverage Curve): {aurc:.4f}")
    # print(f"AUROC (Area Under ROC Curve): {auroc:.4f}")
    # print(f"FPR@95TPR (False Positive Rate at 95% True Positive Rate): {fpr95:.4f}")



if __name__ =='__main__':
    train()