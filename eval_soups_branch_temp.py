import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
import glob
import torch
import torch.nn as nn
import argparse
import numpy as np
from torch.cuda.amp.autocast_mode import autocast
from util import read_conf, validation_accuracy, validate, evaluate, calculate_ece, calculate_nll, validation_accuracy_lora, reliability_diagram
from util.temperature_scaling import ModelWithTemperature
from util.bin_temperature_scaling import ModelWithBinwiseTemperature
import dino_variant
from data import dataloader
import rein
import adaptformer
import torch.nn.functional as F

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

# Model forward function
def temp_forward(model, inputs, temp=1.0, post_temp=False):
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

def initialize_model(variant, config, device, args):
    model_load = dino_variant._small_dino
    dino = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = dino.state_dict()
    
    
    if args.type == 'linear':
        model = dino
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        
    elif args.type == 'rein':
        model = rein.ReinsDinoVisionTransformer(**variant)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        
    elif args.type == 'lora':
        new_state_dict = dict()
        for k in dino_state_dict.keys():
            new_k = k.replace("attn.qkv", "attn.qkv.qkv")
            new_state_dict[new_k] = dino_state_dict[k]
        model = rein.LoRADinoVisionTransformer(dino)
        model.dino.load_state_dict(new_state_dict, strict=False)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)
        # model.to(device)  
        
        
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
        dst_weight = F.interpolate(
            src_weight.float(), size=16, align_corners=False, mode='bilinear') # base model -> 16
        dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
        dst_weight = dst_weight.to(src_weight.dtype)
        new_state_dict['pos_embed'] = torch.cat((extra_tokens, dst_weight), dim=1)
        model = adaptformer.VisionTransformer(patch_size=14, embed_dim= 384, tuning_config = tuning_config, use_dinov2=True)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        # model.load_state_dict(new_state_dict, strict=False) 

    # --------------------------------------------------------------------
    # (A) 모델 전체 state_dict 불러옴 (아직은 랜덤 초기화 파라미터 포함)
    model_dict = model.state_dict()

    # (B) DINO state_dict 중 현재 모델 키/shape와 일치하는 항목만 filtering
    filtered_dict = {}
    for k, v in dino_state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v

    # (C) 모델 dict에 DINO 파라미터를 덮어씌움
    model_dict.update(filtered_dict)

    # (D) strict=True로 최종 로딩 (filtered_dict 외 키는 그대로)
    model.load_state_dict(model_dict, strict=True)
    # --------------------------------------------------------------------
    model.to(device)
    
    return model

def get_model_from_sd(state_dict, variant, config, device, args):
    model_load = dino_variant._small_dino
    dino = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = dino.state_dict()
    
    if args.type == 'linear':
        model = dino
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.load_state_dict(state_dict, strict=True)
        
    elif args.type == 'rein':
        model = rein.ReinsDinoVisionTransformer(**variant)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.load_state_dict(state_dict, strict=True)
        
    elif args.type == 'lora':
        model_load = dino_variant._small_dino
        dino = torch.hub.load('facebookresearch/dinov2', model_load)
        dino_state_dict = dino.state_dict()
        new_state_dict = dict()
        for k in dino_state_dict.keys():
            new_k = k.replace("attn.qkv", "attn.qkv.qkv")
            new_state_dict[new_k] = dino_state_dict[k]
        model = rein.LoRADinoVisionTransformer(dino)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.load_state_dict(state_dict, strict=True)
        
    elif args.type == 'adaptformer':
         # model_load = dino_variant._small_dino
         # dino = torch.hub.load('facebookresearch/dinov2', model_load)
         # dino_state_dict = dino.state_dict()
         
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
         
         model = adaptformer.VisionTransformer(patch_size=14, embed_dim= 384, tuning_config = tuning_config, use_dinov2=True)
         model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
         model.load_state_dict(state_dict, strict=False) 
    model.to(device)

    return model
        


# Greedy soup model ensembling
def greedy_soup_ece(models, model_names, valid_loader, device, variant, config, args):
    # Calculate ECE for each model and sort them by ECE in ascending order (lower ECE is better)
    ece_list = [validate(model, valid_loader, device, args) for model in models]
    # print("ECE for each model:")
    # print(ece_list)
    model_ece_pairs = [(model, ece, name) for model, ece, name in zip(models, ece_list, model_names)]
    sorted_models = sorted(model_ece_pairs, key=lambda x: x[1])
    
    print("Sorted models with ECE performance:")
    for model, ece, name in sorted_models:
        print(f'Model: {name}, ECE: {ece}')

    best_ece = sorted_models[0][1]
    greedy_soup_params = sorted_models[0][0].state_dict()
    greedy_soup_ingredients = [sorted_models[0][0]]
    
    TOLERANCE = (sorted_models[-1][1] - sorted_models[0][1]) / 2
    TOLERANCE = 1
    print(f'Tolerance: {TOLERANCE}')

    for i in range(1, len(models)):
        new_ingredient_params = sorted_models[i][0].state_dict()
        num_ingredients = len(greedy_soup_ingredients)
        print(f'Adding ingredient {i+1} ({sorted_models[i][2]}) to the greedy soup. Num ingredients: {num_ingredients}')
        
        # Calculate potential new parameters with the new ingredient
        potential_greedy_soup_params = {
            k: greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1)) + 
               new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
            for k in new_ingredient_params
        }

        temp_model = get_model_from_sd(potential_greedy_soup_params, variant, config, device, args)
        temp_model.eval()
        temp_model.to(device)
        
        # Evaluate the potential greedy soup model
        outputs, targets = [], []
        with torch.no_grad():
            for inputs, target in valid_loader:
                inputs, target = inputs.to(device), target.to(device)
                if args.type == 'linear':  
                    output = temp_model(inputs)
                    output = torch.softmax(output, dim=1)
                elif args.type == 'rein':
                    output = rein_forward(temp_model, inputs)
                    # print(output.shape)  
                elif args.type == 'lora':
                    with autocast(enabled=True):
                        output = lora_forward(temp_model, inputs)
                elif args.type == 'adaptformer':
                    output = adaptformer_forward(temp_model, inputs)
        
                outputs.append(output.cpu())
                targets.append(target.cpu())
        outputs = torch.cat(outputs).numpy()
        targets = torch.cat(targets).numpy().astype(int)
        held_out_val_ece = calculate_ece(outputs, targets)
        
        print(f'Potential greedy soup ECE: {held_out_val_ece}, best ECE so far: {best_ece}.')
        
        # Add new ingredient to the greedy soup if it improves ECE or is within tolerance
        if held_out_val_ece < best_ece + TOLERANCE:
            best_ece = held_out_val_ece
            greedy_soup_ingredients.append(sorted_models[i][0])
            greedy_soup_params = potential_greedy_soup_params
            print(f'<Added new ingredient to soup. Total ingredients: {len(greedy_soup_ingredients)}>\n')
        else:
            print(f'<No improvement. Reverting to best-known parameters.>\n')


    final_model = get_model_from_sd(greedy_soup_params, variant, config, device, args)
        
    return greedy_soup_params, final_model


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cub')
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--type', '-t', default='rein', type=str)
    parser.add_argument('--checkpoint', '-c', type=str, default='reins_hydra_10')
    args = parser.parse_args()

    config = read_conf(os.path.join('conf', 'data', f'{args.data}.yaml'))
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    checkpoint = args.checkpoint
    
    
    checkpoint_dir = os.path.join(config['save_path'], checkpoint)
    save_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "cyclic_checkpoint_epoch*.pth")))
    
    model_names = [os.path.basename(path) for path in save_paths]

    variant = dino_variant._small_variant
    models = []

    for save_path in save_paths:
        model = initialize_model(variant, config, device, args)
        state_dict = torch.load(save_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        models.append(model)

    
    # models = initialize_models(save_paths, variant, config, device, args)
    train_loader, valid_loader, test_loader = dataloader.setup_data_loaders(args, data_path, batch_size)
    
    greedy_soup_params, model = greedy_soup_ece(models, model_names, valid_loader, device, variant, config, args)

    model = get_model_from_sd(greedy_soup_params, variant, config, device, args)
    
    model_temp = ModelWithTemperature(model)
    model_temp.set_temperature(valid_loader, cross_validate='ece', args=args)
    temp = model_temp.get_temperature()
    print(f"Optimal Temperature: {temp}")

    # model_temp = ModelWithBinwiseTemperature(model, n_bins=10, device='cuda:5')
    # model_temp.set_temperature(valid_loader)
    
    
    ## validation 
    if args.type == 'lora':
        test_accuracy = validation_accuracy_lora(model_temp, test_loader, device)
    else:
        test_accuracy = validation_accuracy(model_temp, test_loader, device, mode=args.type)
    print('test acc:', test_accuracy)

    outputs, targets = [], []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            if args.type == 'rein':
                output = temp_forward(model_temp, inputs, temp=temp, post_temp=True)
                # output = rein_forward(model_temp, inputs)
                output = torch.softmax(output, dim=1)
                # print(output.shape)  
            elif args.type == 'lora':
                with autocast(enabled=True):
                    features = model_temp.forward_features(inputs)
                    output = model_temp.linear(features)
                    output = torch.softmax(output, dim=1)
                    # print(output.shape)
            elif args.type == 'adaptformer':
                output = adaptformer_forward(model_temp, inputs)
                output = torch.softmax(output, dim=1)
                
                
            outputs.append(output.cpu())
            targets.append(target.cpu())
    
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy().astype(int)
    evaluate(outputs, targets, verbose=True)
    
    reliability_diagram(outputs, targets, save_path='reliability_ece_branch_ts.png', title="ECE based")

if __name__ == '__main__':
    train()
