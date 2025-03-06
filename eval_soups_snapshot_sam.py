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
from utils import read_conf, validation_accuracy, ModelWithTemperature, validate, evaluate, calculate_ece, calculate_nll, validation_accuracy_lora, compute_aurc, compute_auroc, compute_fpr95
import dino_variant
from data import dataloader
import rein
from losses import focal_loss

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
            
            
def compute_sharpness(model, dataloader, loss_fn, device, epsilon=1e-3):
    """
    모델의 sharpness를 측정하는 함수.
    - epsilon: perturbation 크기
    """
    sharpness_scores = []
    model.eval()
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 기존 gradient 저장
        for param in model.parameters():
            if param.requires_grad:
                param.grad = None
        
        # Loss 계산 및 backward
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        # 기존 weight 저장 및 perturbation 적용
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        for param in model.parameters():
            if param.requires_grad:
                param.data += epsilon * param.grad.sign()

        # Perturbed 모델에서 다시 loss 계산
        perturbed_outputs = model(inputs)
        perturbed_loss = loss_fn(perturbed_outputs, targets)
        
        # Sharpness 값 저장
        sharpness_score = (perturbed_loss - loss).item()
        sharpness_scores.append(sharpness_score)

        # 원래 weight 복원
        for name, param in model.named_parameters():
            param.data = original_params[name].data

    return sum(sharpness_scores) / len(sharpness_scores)  # 평균 sharpness 값 반환  
            
def frobenius_distance(model1, model2):
    """
    두 모델의 Frobenius Norm Distance 계산
    """
    distance = 0.0
    for (param1, param2) in zip(model1.parameters(), model2.parameters()):
        distance += torch.norm(param1 - param2, p='fro').item()
    return distance

def greedy_soup_weighted(models, model_names, valid_loader, device, variant, config, args):
    """
    Sharpness 기반 가중 평균을 적용한 Greedy Model Soup.
    """
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = focal_loss.FocalLoss(gamma=3)
    
    # 모델별 sharpness 측정
    sharpness_scores = [compute_sharpness(model, valid_loader, loss_fn, device) for model in models]    

    
    # 모델 간 Frobenius 거리 측정
    num_models = len(models)
    distance_matrix = np.zeros((num_models, num_models))

    for i in range(num_models):
        for j in range(i + 1, num_models):
            distance_matrix[i, j] = frobenius_distance(models[i], models[j])
            distance_matrix[j, i] = distance_matrix[i, j]

    # 모델들을 Clustering하여 비슷한 Basin끼리 묶음
    from sklearn.cluster import AgglomerativeClustering
    
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0, affinity='precomputed', linkage='average')
    cluster_labels = clustering.fit_predict(distance_matrix)

    # Greedy Soup 초기화 (가장 좋은 모델을 기준으로 시작)
    best_model_idx = np.argmin(sharpness_scores)  # Sharpness가 가장 낮은 (flat한) 모델 선택
    best_model = models[best_model_idx]
    greedy_soup_params = best_model.state_dict()
    
    # Sharpness 기반 가중치 계산
    lambda_val = 0.5  # Hyperparameter (Sharpness에 대한 감도 조절)
    weights = np.exp(-lambda_val * np.array(sharpness_scores))
    weights /= np.sum(weights)  # 정규화

    print("Applying weighted averaging with the following weights:")
    print(weights)

    # 가중 평균 수행
    weighted_avg_params = {k: torch.zeros_like(v) for k, v in greedy_soup_params.items()}

    for i, model in enumerate(models):
        model_params = model.state_dict()
        for k in model_params:
            weighted_avg_params[k] += weights[i] * model_params[k]

    # 최종 Model 생성
    final_model = get_model_from_sd(weighted_avg_params, variant, config, device, args)
    final_model.to(device)
    
    return weighted_avg_params, final_model



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cifar100')
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--type', '-t', default='rein', type=str)
    parser.add_argument('--checkpoint', '-c', type=str)
    # parser.add_argument('--soup', '-s', type=str, default='acc')
    args = parser.parse_args()

    config = read_conf(os.path.join('conf', 'data', f'{args.data}.yaml'))
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    checkpoint = args.checkpoint
    num_workers = int(config['num_workers'])
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
    _, valid_loader, test_loader = dataloader.setup_data_loaders(args, data_path, batch_size)
    
    print('Greedy soup')
    greedy_soup_params, model = greedy_soup_weighted(models, model_names, valid_loader, device, variant, config, args)

    

    model = get_model_from_sd(greedy_soup_params, variant, config, device, args)
    model.eval()
    

    ## validation 
    if args.type == 'lora':
        test_accuracy = validation_accuracy_lora(model, test_loader, device)
    else:
        test_accuracy = validation_accuracy(model, test_loader, device, mode=args.type)
    print("\n🔹 Model Accuracy 🔹")
    print('Test Acc:', test_accuracy)

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
    # Failure Prediction Metrics 계산
    # aurc = compute_aurc(outputs, targets)
    auroc = compute_auroc(outputs, targets)
    fpr95 = compute_fpr95(outputs, targets)
    
    print("\n🔹 Failure Prediction Metrics 🔹")
    # print(f"AURC (Area Under Risk-Coverage Curve): {aurc:.4f}")
    print(f"AUROC (Area Under ROC Curve): {auroc:.4f}")
    print(f"FPR@95TPR (False Positive Rate at 95% True Positive Rate): {fpr95:.4f}")

if __name__ == '__main__':
    train()
