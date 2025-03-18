import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
import torch
import torch.nn as nn
import argparse
import numpy as np

import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from utils import read_conf, validation_accuracy, ModelWithTemperature, validate, evaluate, calculate_ece, calculate_nll, validation_accuracy_lora
import dino_variant
from data import dataloader
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




def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    true = np.array(true).reshape(-1)
    pred = np.array(pred).reshape(-1)
    conf = np.array(conf).reshape(-1)

    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0, 0, 0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])
        len_bin = len(filtered_tuples)
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin
        accuracy = float(correct) / len_bin
        return accuracy, avg_conf, len_bin

def ECE(conf, pred, true, bin_size=0.1):
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    n = len(conf)
    ece = 0
    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        ece += np.abs(acc - avg_conf) * len_bin / n
    return ece

def OE(conf, pred, true, bin_size = 0.1):
    
    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        ece: expected calibration error
    """
    true = np.array(true).reshape(-1)  
    pred = np.array(pred).reshape(-1)
    conf = np.array(conf).reshape(-1)

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)  
        # print(acc, avg_conf, len_bin)
        if avg_conf > acc:
            ece += avg_conf * (avg_conf - acc) * len_bin / n 
        
    return ece


def evaluate_ensemble(outputs, targets, bins=15):
    probs = np.array(outputs)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    
    ece = ECE(confs, preds, targets, bin_size=1/bins)
    oe = OE(confs, preds, targets, bin_size = 1/bins)
    
    
    print(f"ECE: {ece:.4f}")
    print(f"OE: {oe:.4f}")
    print("==========================")

def ensemble_evaluate(models, test_loader, device, args):
    outputs = []
    targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            
            if target.ndim > 1 and target.size(1) > 1:
                target = torch.argmax(target, dim=1)
            if target.ndim > 1:
                target = target.view(-1)
                
            batch_outputs = []
            
            if args.type == 'rein':
                for model in models:
                    output = rein_forward(model, inputs)
                    batch_outputs.append(output)
            elif args.type == 'lora':
                for model in models:
                    output = lora_forward(model, inputs)
                    batch_outputs.append(output)

            ensemble_output = torch.stack(batch_outputs).mean(dim=0)
            outputs.append(ensemble_output.cpu().numpy())
            targets.append(target.cpu().numpy())
            

            _, predicted = torch.max(ensemble_output, dim=1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    accuracy = correct / total * 100
    print("=== Evaluation Results ===")
    print(f"Ensemble Accuracy: {accuracy:.2f}%")

    outputs = np.vstack(outputs)
    targets = np.concatenate(targets)
    evaluate_ensemble(outputs, targets)

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
        
    train_loader, valid_loader, test_loader = dataloader.setup_data_loaders(args, data_path, batch_size)
    
    ensemble_evaluate(models, test_loader, device, args)
    

if __name__ == '__main__':
    train()
