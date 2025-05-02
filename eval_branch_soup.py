import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import contextlib
import io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import numpy as np
import glob
from torch.cuda.amp.autocast_mode import autocast
from util import read_conf, validation_accuracy, ModelWithTemperature, validate, evaluate, calculate_ece, calculate_nll, validation_accuracy_lora, compute_aurc, compute_auroc, compute_fpr95, ece, sce, ace, tace, reliability_diagram
import dino_variant
from data import dataloader
import rein
import adaptformer


# Model forward function
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

  

def initialize_model(variant, config, device, args):
    model_load = dino_variant._small_dino
    dino = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = dino.state_dict()
    
    
    if args.adapter == 'linear':
        model = dino
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        
    elif args.adapter == 'rein':
        model = rein.ReinsDinoVisionTransformer(**variant)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        
    elif args.adapter == 'lora':
        new_state_dict = dict()
        for k in dino_state_dict.keys():
            new_k = k.replace("attn.qkv", "attn.qkv.qkv")
            new_state_dict[new_k] = dino_state_dict[k]
        model = rein.LoRADinoVisionTransformer(dino)
        model.dino.load_state_dict(new_state_dict, strict=False)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)
        # model.to(device)  
        
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
    
    if args.adapter == 'linear':
        model = dino
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.load_state_dict(state_dict, strict=True)
        
    elif args.adapter == 'rein':
        model = rein.ReinsDinoVisionTransformer(**variant)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.load_state_dict(state_dict, strict=True)
        
    elif args.adapter == 'lora':
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
        
    elif args.adapter == 'adaptformer':
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

    return model

def uniform_soup(models, model_names, device, variant, config, args):
    print("Creating uniform soup from models (no evaluation).")
    
    num_models = len(models)

    # 첫 번째 모델에서 시작
    for i, model in enumerate(models):
        state_dict = model.state_dict()
        if i == 0:
            uniform_soup_params = {k: v.clone() * (1. / num_models) for k, v in state_dict.items()}
        else:
            for k in uniform_soup_params:
                uniform_soup_params[k] += state_dict[k].clone() * (1. / num_models)

        print(f"Added model {model_names[i]} to uniform soup ({i+1}/{num_models}).")

    final_model = get_model_from_sd(uniform_soup_params, variant, config, device, args)

    return uniform_soup_params, final_model

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
    soup_params = sorted_models[0][0].state_dict()
    greedy_soup_ingredients = [sorted_models[0][0]]
    
    TOLERANCE = (sorted_models[-1][1] - sorted_models[0][1]) / 2
    TOLERANCE = 1
    print(f'Tolerance: {TOLERANCE}')

    for i in range(1, len(models)):
        new_ingredient_params = sorted_models[i][0].state_dict()
        num_ingredients = len(greedy_soup_ingredients)
        print(f'Adding ingredient {i+1} ({sorted_models[i][2]}) to the greedy soup. Num ingredients: {num_ingredients}')
        
        # Calculate potential new parameters with the new ingredient
        potential_soup_params = {
            k: soup_params[k].clone() * (num_ingredients / (num_ingredients + 1)) + 
               new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
            for k in new_ingredient_params
        }

        temp_model = get_model_from_sd(potential_soup_params, variant, config, device, args)
        temp_model.eval()
        temp_model.to(device)
        
        # Evaluate the potential greedy soup model
        outputs, targets = [], []
        with torch.no_grad():
            for inputs, target in valid_loader:
                inputs, target = inputs.to(device), target.to(device)
                if args.adapter == 'linear':  
                    output = temp_model(inputs)
                    output = torch.softmax(output, dim=1)
                elif args.adapter == 'rein':
                    output = rein_forward(temp_model, inputs)
                    # print(output.shape)  
                elif args.adapter == 'lora':
                    with autocast(enabled=True):
                        output = lora_forward(temp_model, inputs)
                elif args.adapter == 'adaptformer':
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
            soup_params = potential_soup_params
            print(f'<Added new ingredient to soup. Total ingredients: {len(greedy_soup_ingredients)}>\n')
        else:
            print(f'<No improvement. Reverting to best-known parameters.>\n')


    final_model = get_model_from_sd(soup_params, variant, config, device, args)
        
    return soup_params, final_model


def greedy_soup_acc(models, model_names, valid_loader, device, variant, config, args):
    # Evaluate and sort models by validation accuracy
    if args.adapter == 'rein' or args.adapter == 'adaptformer':
        model_accuracies = [(model, validation_accuracy(model, valid_loader, device, mode=args.adapter), name) for model, name in zip(models, model_names)]
    elif args.adapter == 'lora':
        model_accuracies = [(model, validation_accuracy_lora(model, valid_loader, device), name) for model, name in zip(models, model_names)]

    
    # Sort models based on accuracy
    sorted_models = sorted(model_accuracies, key=lambda x: x[1], reverse=True)
    
    # Print sorted models with their names and accuracies
    print("Sorted models by accuracy:")
    for model, acc, name in sorted_models:
        print(f'Model: {name}, Accuracy: {acc}')
    print("\n")
    
    # Initialize greedy soup with the highest-performing model
    max_accuracy = sorted_models[0][1]
    soup_params = sorted_models[0][0].state_dict()  # Best model's initial parameters
    greedy_soup_ingredients = [sorted_models[0][0]] 

    for i in range(1, len(sorted_models)):
        print(f'Testing model {i+1} ({sorted_models[i][2]}) of {len(sorted_models)}')
        
        # previous_soup_params = {k: v.clone() for k, v in soup_params.items()}
        
        # New model parameters to test as an additional ingredient
        new_ingredient_params = sorted_models[i][0].state_dict()
        num_ingredients = len(greedy_soup_ingredients)
        print(f'Adding ingredient {i+1} ({sorted_models[i][2]}) to the greedy soup. Num ingredients: {num_ingredients}')    
    
        # Create potential new soup parameters by averaging with the new ingredient
        potential_soup_params = {
            k: soup_params[k].clone() * (num_ingredients / (num_ingredients + 1)) +
               new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
            for k in new_ingredient_params
        }
        
        # Load the new potential parameters into the base model for evaluation
        temp_model = get_model_from_sd(potential_soup_params, variant, config, device, args)
        temp_model.to(device)
        temp_model.eval()
        
        # Calculate validation accuracy with the potential new soup parameters
        if args.adapter == 'linear' or args.adapter == 'rein' or args.adapter == 'adaptformer':
            held_out_val_accuracy = validation_accuracy(temp_model, valid_loader, device, mode=args.adapter)
        elif args.adapter == 'lora':
            held_out_val_accuracy = validation_accuracy_lora(temp_model, valid_loader, device)
        elif args.adapter == 'adaptformer':
            held_out_val_accuracy = validation_accuracy(temp_model, valid_loader, device, mode=args.adapter)

        
        print(f'Held-out validation accuracy: {held_out_val_accuracy}, best accuracy so far: {max_accuracy}.\n')
        
        # Update greedy soup if accuracy improves, otherwise revert to original parameters
        if held_out_val_accuracy > max_accuracy:
            greedy_soup_ingredients.append(sorted_models[i][0])
            max_accuracy = held_out_val_accuracy
            soup_params = potential_soup_params  # Save the improved parameters
            print(f'[New greedy soup ingredient added. Number of ingredients: {len(greedy_soup_ingredients)}]\n')
        else:
            print(f'[No improvement. Reverting to best-known parameters.]\n')
         
        final_model = get_model_from_sd(soup_params, variant, config, device, args)
        

    return soup_params, final_model


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='eyepacs')
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--adapter', '-a', default='rein', type=str)
    parser.add_argument('--checkpoint', '-c', type=str, default='reins_hydra_10')
    parser.add_argument('--soup', '-s', type=str, default='uniform')
    parser.add_argument('--save_file', '-n', type=str, default='branch_soup')
    args = parser.parse_args()

    config = read_conf(os.path.join('conf', 'data', f'{args.data}.yaml'))
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    checkpoint = args.checkpoint
    # num_workers = int(config['num_workers'])
   
    checkpoint_dir = os.path.join(config['save_path'], checkpoint)
    save_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "cyclic_checkpoint_epoch*.pth")))

    # print(save_paths) 
    print(f'Found {len(save_paths)} models to soup.')
    
    
    model_names = [os.path.basename(path) for path in save_paths]

    variant = dino_variant._small_variant
    
    models = []

    
    for save_path in save_paths:
        if args.adapter == 'adaptformer':
            model = initialize_model(variant, config, device, args)
            state_dict= torch.load(save_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            models.append(model)
            
        else:
            model = initialize_model(variant, config, device, args)
            state_dict = torch.load(save_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=True) # 수정
            model.to(device)
            model.eval()
            models.append(model)

    
    # models = initialize_models(save_paths, variant, config, device, args)
    # train_loader, valid_loader, test_loader = dataloader.setup_data_loaders(args, data_path, batch_size)
    train_loader, valid_loader, test_loader = dataloader.setup_data_loaders(args, data_path, batch_size)

    
    if args.soup == 'acc':
        print('Greedy soup by ACC')
        soup_params, model = greedy_soup_acc(models, model_names, valid_loader, device, variant, config, args)
    elif args.soup == 'ece':
        print('Greedy soup by ECE')
        soup_params, model = greedy_soup_ece(models, model_names, valid_loader, device, variant, config, args)
    elif args.soup == 'uniform':
        print('Uniform soup')
        soup_params, model = uniform_soup(models, model_names, device, variant, config, args)
    

    model = get_model_from_sd(soup_params, variant, config, device, args)
    model.eval()
    model.to(device)
    
    save_dir = os.path.join(config['save_path'], 'branch_soup')
    os.makedirs(save_dir, exist_ok=True)
    
    # ckpt_name = args.save_file + '.pth'
    # save_path = os.path.join(save_dir, ckpt_name)

    # torch.save(model.state_dict(), save_path)
    # print(f"\nBranch Soup parameter saved to '{save_path}'")


    ## validation 
    if args.adapter == 'lora':
        test_accuracy = validation_accuracy_lora(model, test_loader, device)
    else:
        test_accuracy = validation_accuracy(model, test_loader, device, mode=args.adapter)
    print("\n🔹 Model Accuracy 🔹")
    print('Test Acc:', test_accuracy)

    outputs, targets = [], []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            if args.adapter == 'linear':
                output = model(inputs)
                output = model.linear(output)
                output = torch.softmax(output, dim=1)
            elif args.adapter == 'rein':
                output = rein_forward(model, inputs)
                # print(output.shape)  
            elif args.adapter == 'lora':
                with autocast(enabled=True):
                    features = model.forward_features(inputs)
                    output = model.linear(features)
                    output = torch.softmax(output, dim=1)
                    # print(output.shape)
            elif args.adapter == 'adaptformer':
                output = adaptformer_forward(model, inputs)

            outputs.append(output.cpu())
            targets.append(target.cpu())
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy().astype(int)
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
    
    
    # reliability_diagram(outputs, targets, num_bins=15, title="ECE based", save_path="reliability_ece_branch.png")

    # reliability_diagram(probs=outputs, labels=targets, num_bins=15, threshold=0.0, title="ACE based (All probs)")

    # reliability_diagram(probs=outputs, labels=targets, num_bins=15, threshold=0.01, title="TACE based (Threshold=0.01)")
    # Failure Prediction Metrics 계산
    # aurc = compute_aurc(outputs, targets)
    # auroc = compute_auroc(outputs, targets)
    # fpr95 = compute_fpr95(outputs, targets)
    
    # print("\n🔹 Failure Prediction Metrics 🔹")
    # print(f"AURC (Area Under Risk-Coverage Curve): {aurc:.4f}")
    # print(f"AUROC (Area Under ROC Curve): {auroc:.4f}")
    # print(f"FPR@95TPR (False Positive Rate at 95% True Positive Rate): {fpr95:.4f}")

if __name__ == '__main__':
    train()
