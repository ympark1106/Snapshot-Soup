import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
import torch
import torch.nn as nn
import argparse
import numpy as np

from utils import read_conf, validation_accuracy, ModelWithTemperature, validate, evaluate, calculate_ece, calculate_nll
import dino_variant
from data import cifar10, cifar100, cub, ham10000, bloodmnist
import rein

# Model forward function
def rein_forward(model, inputs, temp_scaler=None):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    if temp_scaler:
        output = temp_scaler.temperature_scale(output)
    output = torch.softmax(output, dim=1)
    return output

def initialize_models(save_paths, variant, config, device):
    models = []
    for save_path in save_paths:
        model = rein.ReinsDinoVisionTransformer(**variant)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        state_dict = torch.load(save_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        models.append(model)
    return models

# Data loader setup
def setup_data_loaders(args, data_path, batch_size):
    if args.data == 'cifar10':
        test_loader = cifar10.get_test_loader(batch_size, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
        valid_loader = None
    elif args.data == 'cifar100':
        test_loader = cifar100.get_test_loader(data_dir=data_path, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        _, valid_loader = cifar100.get_train_valid_loader(data_dir=data_path, augment=True, batch_size=32, valid_size=0.1, random_seed=42, shuffle=True, num_workers=4, pin_memory=True)
    elif args.data == 'cub':
        test_loader = cub.get_test_loader(data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=4, pin_memory=True)
        _, valid_loader = cub.get_train_val_loader(data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=4, pin_memory=True)
    elif args.data == 'ham10000':
        _, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=32, num_workers=4)
    elif args.data == 'bloodmnist':
        _, test_loader, valid_loader = bloodmnist.get_dataloader(batch_size,download=True, num_workers=4)
    else:
        raise ValueError(f"Unsupported data type: {args.data}")
    
    return test_loader, valid_loader

def get_model_from_sd(state_dict, variant, config, device):
    model = rein.ReinsDinoVisionTransformer(**variant)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    
    return model


# Validation and test accuracy calculation
def validate_model(model, valid_loader, device, mode):
    return validation_accuracy(model, valid_loader, device, mode=mode)

# Sort models by accuracy
def sort_models_by_accuracy(models, valid_loader, device, mode):
    model_accuracies = [(model, validate_model(model, valid_loader, device, mode)) for model in models]
    sorted_models = sorted(model_accuracies, key=lambda x: x[1], reverse=True)  # Sort by accuracy (descending)
    return sorted_models

# Greedy soup ensemble function
def greedy_soup_ensemble(models, model_names, valid_loader, device, variant, config):
    # Evaluate and sort models by validation accuracy
    model_accuracies = [(model, validation_accuracy(model, valid_loader, device), name) for model, name in zip(models, model_names)]
    
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
        temp_model = get_model_from_sd(potential_greedy_soup_params, variant, config, device)
        temp_model.eval()
        
        # Calculate validation accuracy with the potential new soup parameters
        held_out_val_accuracy = validation_accuracy(temp_model, valid_loader, device)
        
        print(f'Held-out validation accuracy: {held_out_val_accuracy}, best accuracy so far: {max_accuracy}.\n')
        
        # Update greedy soup if accuracy improves, otherwise revert to original parameters
        if held_out_val_accuracy > max_accuracy:
            greedy_soup_ingredients.append(sorted_models[i][0])
            max_accuracy = held_out_val_accuracy
            greedy_soup_params = potential_greedy_soup_params  # Save the improved parameters
            print(f'[New greedy soup ingredient added. Number of ingredients: {len(greedy_soup_ingredients)}]\n')
        else:
            print(f'[No improvement. Reverting to best-known parameters.]\n')

    final_model = get_model_from_sd(greedy_soup_params, variant, config, device)
        
    return greedy_soup_params, final_model




# Final evaluation
def evaluate_model(model, test_loader, device):
    outputs, targets = [], []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = rein_forward(model, inputs)
            outputs.append(output.cpu())
            targets.append(target.cpu())

    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy().astype(int)
    evaluate(outputs, targets, verbose=True)


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

        # os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch99.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch129.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch159.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch189.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch219.pth'),
        
        os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch99.pth'),
        os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch129.pth'),
        os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch159.pth'),
        os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch189.pth'),
        os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch219.pth'),
    
        # os.path.join(config['save_path'], 'reins_focal_hydra_2/cyclic_checkpoint_epoch89.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_2/cyclic_checkpoint_epoch129.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_2/cyclic_checkpoint_epoch169.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_2/cyclic_checkpoint_epoch209.pth'),
    ]
    
    model_names = [os.path.basename(path) for path in save_paths]

    variant = dino_variant._small_variant
    models = initialize_models(save_paths, variant, config, device)
    test_loader, valid_loader = setup_data_loaders(args, data_path, batch_size)
    
    # Step 1: Compute greedy soup parameters
    greedy_soup_params, model1 = greedy_soup_ensemble(models, model_names, valid_loader, device, variant, config)

    # Evaluate the final model on the test set
    model1.load_state_dict(greedy_soup_params)
    model1.eval()
    test_accuracy = validation_accuracy(model1, test_loader, device, mode=args.type)
    print('Test accuracy:', test_accuracy)

    outputs, targets = [], []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = rein_forward(model1, inputs)
            outputs.append(output.cpu())
            targets.append(target.cpu())
    
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy().astype(int)
    evaluate(outputs, targets, verbose=True)

if __name__ == '__main__':
    train()