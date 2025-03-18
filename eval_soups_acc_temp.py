import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from torch.cuda.amp.autocast_mode import autocast
from utils import read_conf, validation_accuracy, validate, evaluate, calculate_ece, calculate_nll, validation_accuracy_lora
from utils.temperature_scaling import ModelWithTemperature
import dino_variant
from data import dataloader
import rein
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

    if args.type == 'rein':
        model = rein.ReinsDinoVisionTransformer(
            **variant
        )
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.load_state_dict(dino_state_dict, strict=False)
        model.to(device)

    elif args.type == 'lora':
        new_state_dict = dict()
        for k in dino_state_dict.keys():
            new_k = k.replace("attn.qkv", "attn.qkv.qkv")
            new_state_dict[new_k] = dino_state_dict[k]
        model = rein.LoRADinoVisionTransformer(dino)
        model.dino.load_state_dict(new_state_dict, strict=False)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
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


# Greedy soup model ensembling
def greedy_soup_ensemble(models, model_names, valid_loader, device, variant, config, args):
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

    return greedy_soup_params, sorted_models[0][0]


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cub')
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--type', '-t', default='rein', type=str)
    args = parser.parse_args()

    config = read_conf(os.path.join('conf', 'data', f'{args.data}.yaml'))
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    
    save_paths = [
        
        os.path.join(config['save_path'], 'reins_focal_1'),
        os.path.join(config['save_path'], 'reins_focal_2'),
        os.path.join(config['save_path'], 'reins_focal_3'),
        os.path.join(config['save_path'], 'reins_focal_4'),
        os.path.join(config['save_path'], 'reins_focal_5'),        
        os.path.join(config['save_path'], 'reins_focal_6'),
        os.path.join(config['save_path'], 'reins_focal_7'),
        os.path.join(config['save_path'], 'reins_focal_8'),
        os.path.join(config['save_path'], 'reins_focal_9'),
        os.path.join(config['save_path'], 'reins_focal_10'),
        
    ]
    
    model_names = [os.path.basename(path) for path in save_paths]

    variant = dino_variant._small_variant
    models = []

    for save_path in save_paths:
        model = initialize_model(variant, config, device, args)
        state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        models.append(model)

    
    # models = initialize_models(save_paths, variant, config, device, args)
    train_loader, valid_loader, test_loader = dataloader.setup_data_loaders(args, data_path, batch_size)
    
    greedy_soup_params, model = greedy_soup_ensemble(models, model_names, valid_loader, device, variant, config, args)

    model = get_model_from_sd(greedy_soup_params, variant, config, device, args)
    
    model_temp = ModelWithTemperature(model)
    # print(model_temp)
    model_temp.set_temperature(valid_loader, cross_validate='ece')
    temp = model_temp.get_temperature()
    print(f"Optimal Temperature: {temp}")
    
    
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
                output = torch.softmax(output, dim=1)
                # print(output.shape)  
            elif args.type == 'lora':
                with autocast(enabled=True):
                    features = model_temp.forward_features(inputs)
                    output = model_temp.linear(features)
                    output = torch.softmax(output, dim=1)
                    # print(output.shape)
                
                
            outputs.append(output.cpu())
            targets.append(target.cpu())
    
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy().astype(int)
    evaluate(outputs, targets, verbose=True)

if __name__ == '__main__':
    train()
