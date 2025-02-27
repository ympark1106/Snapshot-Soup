import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import contextlib
import io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn
import argparse
import numpy as np
import glob
from torch.cuda.amp.autocast_mode import autocast
from utils import read_conf, validation_accuracy, ModelWithTemperature, validate, evaluate, calculate_ece, calculate_nll, validation_accuracy_lora
import dino_variant
from data import cifar10, cifar100, ham10000
import rein
from losses import DECE

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

# def initialize_model(variant, config, device, args):
#     model_load = dino_variant._small_dino
#     dino = torch.hub.load('facebookresearch/dinov2', model_load)
#     dino_state_dict = dino.state_dict()

#     if args.type == 'rein':
#         model = rein.ReinsDinoVisionTransformer(
#             **variant
#         )
#         model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
#         model.load_state_dict(dino_state_dict, strict=True) # 수정
#         model.to(device)

#     elif args.type == 'lora':
#         new_state_dict = dict()
#         for k in dino_state_dict.keys():
#             new_k = k.replace("attn.qkv", "attn.qkv.qkv")
#             new_state_dict[new_k] = dino_state_dict[k]
#         model = rein.LoRADinoVisionTransformer(dino)
#         model.dino.load_state_dict(new_state_dict, strict=True)
#         model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
#         model.to(device)
        
#     return model

def initialize_model(variant, config, device, args):
    model_load = dino_variant._small_dino
    dino = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = dino.state_dict()

    # ReinsDinoVisionTransformer 모델 생성
    if args.type == 'rein':
        model = rein.ReinsDinoVisionTransformer(**variant)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    elif args.type == 'lora':
        # LoRA 계열 모델은 attn.qkv를 attn.qkv.qkv로 rename
        new_state_dict = {}
        for k, v in dino_state_dict.items():
            new_k = k.replace("attn.qkv", "attn.qkv.qkv")
            new_state_dict[new_k] = v
        dino_state_dict = new_state_dict

        model = rein.LoRADinoVisionTransformer(dino)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])

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
    if args.type == 'rein':
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
    model.to(device)
    
    return model
            
# Data loader setup
def setup_data_loaders(args, data_path, batch_size):
    if args.data == 'cifar10':
        test_loader = cifar10.get_test_loader(batch_size, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
        valid_loader = None
    elif args.data == 'cifar100':
        test_loader = cifar100.get_test_loader(data_dir=data_path, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        _, valid_loader = cifar100.get_train_valid_loader(data_dir=data_path, augment=True, batch_size=batch_size, valid_size=0.1, random_seed=42, shuffle=True, num_workers=4, pin_memory=True)
    elif args.data == 'ham10000':
        _, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=32, num_workers=4)
    # elif args.data == 'bloodmnist':
    #     _, valid_loader, test_loader = bloodmnist.get_dataloader(batch_size=32, download=True, num_workers=4)
    # elif args.data == 'pathmnist':
    #     _, valid_loader, test_loader = pathmnist.get_dataloader(batch_size=32, download=True, num_workers=4)
    # elif args.data == 'retinamnist':
    #     _, valid_loader, test_loader = retinamnist.get_dataloader(batch_size=32, download=True, num_workers=4)
    else:
        raise ValueError(f"Unsupported data type: {args.data}")
    
    return test_loader, valid_loader

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
    TOLERANCE = 0

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
        
        # Evaluate the potential greedy soup model
        outputs, targets = [], []
        with torch.no_grad():
            for inputs, target in valid_loader:
                inputs, target = inputs.to(device), target.to(device)
                if args.type == 'rein':
                    output = rein_forward(temp_model, inputs)
                    # print(output.shape)  
                elif args.type == 'lora':
                    with autocast(enabled=True):
                        output = lora_forward(temp_model, inputs)
        
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


def greedy_soup_acc(models, model_names, valid_loader, device, variant, config, args):
    # Evaluate and sort models by validation accuracy
    if args.type == 'rein':
        model_accuracies = [(model, validation_accuracy(model, valid_loader, device), name) for model, name in zip(models, model_names)]
    elif args.type == 'lora':
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
    greedy_soup_params = sorted_models[0][0].state_dict()  # Best model's initial parameters
    greedy_soup_ingredients = [sorted_models[0][0]] 

    for i in range(1, len(sorted_models)):
        print(f'Testing model {i+1} ({sorted_models[i][2]}) of {len(sorted_models)}')
        
        # previous_greedy_soup_params = {k: v.clone() for k, v in greedy_soup_params.items()}
        
        # New model parameters to test as an additional ingredient
        new_ingredient_params = sorted_models[i][0].state_dict()
        num_ingredients = len(greedy_soup_ingredients)
        print(f'Adding ingredient {i+1} ({sorted_models[i][2]}) to the greedy soup. Num ingredients: {num_ingredients}')    
    
        # Create potential new soup parameters by averaging with the new ingredient
        potential_greedy_soup_params = {
            k: greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1)) +
               new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
            for k in new_ingredient_params
        }
        
        # Load the new potential parameters into the base model for evaluation
        temp_model = get_model_from_sd(potential_greedy_soup_params, variant, config, device, args)
        temp_model.eval()
        
        # Calculate validation accuracy with the potential new soup parameters
        if args.type == 'rein':
            held_out_val_accuracy = validation_accuracy(temp_model, valid_loader, device, mode='rein')
        elif args.type == 'lora':
            held_out_val_accuracy = validation_accuracy_lora(temp_model, valid_loader, device)
        
        print(f'Held-out validation accuracy: {held_out_val_accuracy}, best accuracy so far: {max_accuracy}.\n')
        
        # Update greedy soup if accuracy improves, otherwise revert to original parameters
        if held_out_val_accuracy > max_accuracy:
            greedy_soup_ingredients.append(sorted_models[i][0])
            max_accuracy = held_out_val_accuracy
            greedy_soup_params = potential_greedy_soup_params  # Save the improved parameters
            print(f'[New greedy soup ingredient added. Number of ingredients: {len(greedy_soup_ingredients)}]\n')
        else:
            print(f'[No improvement. Reverting to best-known parameters.]\n')
         
        final_model = get_model_from_sd(greedy_soup_params, variant, config, device, args)
        

    return greedy_soup_params, final_model


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cifar100')
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--type', '-t', default='rein', type=str)
    parser.add_argument('--checkpoint', '-c', type=str)
    parser.add_argument('--soup', '-s', type=str, default='acc')
    args = parser.parse_args()

    config = read_conf(os.path.join('conf', 'data', f'{args.data}.yaml'))
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    checkpoint = args.checkpoint
    
    save_paths = [ 
        # os.path.join(config['save_path'], checkpoint, 'cyclic_checkpoint_epoch219.pth'),
        # os.path.join(config['save_path'], checkpoint, 'cyclic_checkpoint_epoch249.pth'),
    ]
    
    
    checkpoint_dir = os.path.join(config['save_path'], checkpoint)
    save_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "cyclic_checkpoint_epoch*.pth")))

    # print(save_paths) 
    print(f'Found {len(save_paths)} models to soup.')
    
    
    model_names = [os.path.basename(path) for path in save_paths]

    variant = dino_variant._small_variant
    
    models = []

    
    for save_path in save_paths:
        model = initialize_model(variant, config, device, args)
        state_dict = torch.load(save_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True) # 수정
        model.to(device)
        model.eval()
        models.append(model)

    
    # models = initialize_models(save_paths, variant, config, device, args)
    test_loader, valid_loader = setup_data_loaders(args, data_path, batch_size)
    
    if args.soup == 'acc':
        print('Greedy soup by ACC')
        greedy_soup_params, model = greedy_soup_acc(models, model_names, valid_loader, device, variant, config, args)
    elif args.soup == 'ece':
        print('Greedy soup by ECE')
        greedy_soup_params, model = greedy_soup_ece(models, model_names, valid_loader, device, variant, config, args)
    

    model = get_model_from_sd(greedy_soup_params, variant, config, device, args)
    model.eval()
    

    ## validation 
    if args.type == 'lora':
        test_accuracy = validation_accuracy_lora(model, test_loader, device)
    else:
        test_accuracy = validation_accuracy(model, test_loader, device, mode=args.type)
    print('test acc:', test_accuracy)

    outputs, targets = [], []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            if args.type == 'rein':
                output = rein_forward(model, inputs)
                # print(output.shape)  
            elif args.type == 'lora':
                with autocast(enabled=True):
                    features = model.forward_features(inputs)
                    output = model.linear(features)
                    output = torch.softmax(output, dim=1)
                    # print(output.shape)
                
                
            outputs.append(output.cpu())
            targets.append(target.cpu())
    
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy().astype(int)
    evaluate(outputs, targets, verbose=True)

if __name__ == '__main__':
    train()
