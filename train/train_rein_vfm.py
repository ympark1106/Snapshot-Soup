import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import timm
import numpy as np
import utils

import random
import rein
# import lora
# import open_clip
# import clip
import adaptformer

import dino_variant
from sklearn.metrics import f1_score

def set_requires_grad(model: nn.Module, keywords):
    """
    notice:key in name!
    """
    requires_grad_names = []
    num_params = 0
    num_trainable = 0
    params = []
    for name, param in model.named_parameters():
        num_params += param.numel()
        if any(key in name for key in keywords):
            print(name)
            param.requires_grad = True
            requires_grad_names.append(name)
            num_trainable += param.numel()
            params.append(param)
        else:
            # print(name, 'no_grad')
            param.requires_grad = False
    return params

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--net', default='dinov2', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--noise_rate', '-n', type=float, default=0.2)
    parser.add_argument('--adapter', default='rein', type=str)


    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['id_dataset']
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    noise_rate = args.noise_rate

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch), int(0.9*max_epoch)]

    if args.data == 'ham10000':
        train_loader, valid_loader = utils.get_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)
    elif args.data == 'aptos':
        train_loader, valid_loader = utils.get_aptos_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)
    elif args.data == 'nihchest':
        train_loader, valid_loader = utils.get_nihxray(data_path, batch_size = batch_size)
    elif args.data == 'idrid':
        train_loader, valid_loader = utils.get_idrid_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)
    elif args.data == 'chaoyang':
        train_loader, valid_loader = utils.get_chaoyang_dataset(data_path, batch_size = batch_size)
    elif 'mnist' in args.data:
        train_loader, valid_loader = utils.get_mnist_noise_dataset(args.data, noise_rate=noise_rate, batch_size = batch_size)
    elif args.data == 'dr':
        train_loader, valid_loader = utils.get_dr(data_path, batch_size = batch_size)

    tuning_config = argparse.Namespace()
    if args.adapter == 'adaptformer':
        # Adaptformer
        tuning_config.ffn_adapt = True
        tuning_config.ffn_num = 64
        tuning_config.ffn_option="parallel"
        tuning_config.ffn_adapter_layernorm_option="none"
        tuning_config.ffn_adapter_init_option="lora"
        tuning_config.ffn_adapter_scalar="0.1"
        tuning_config.d_model=768
        # VPT
        tuning_config.vpt_on = False
        tuning_config.vpt_num = 1

        tuning_config.fulltune = False
    elif args.adapter == 'vpt':
        # Adaptformer
        tuning_config.ffn_adapt = False
        tuning_config.ffn_num = 64
        tuning_config.ffn_option="parallel"
        tuning_config.ffn_adapter_layernorm_option="none"
        tuning_config.ffn_adapter_init_option="lora"
        tuning_config.ffn_adapter_scalar="0.1"
        tuning_config.d_model=768
        # VPT
        tuning_config.vpt_on = True
        tuning_config.vpt_num = 10


    if args.net == 'dinov2':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant

        model = torch.hub.load('facebookresearch/dinov2', model_load)
        dino_state_dict = model.state_dict()

        if args.adapter == 'rein':
            model = rein.ReinsDinoVisionTransformer(
                **variant
            )
            new_state_dict = dino_state_dict
        if args.adapter == 'adaptformer' or args.adapter == 'vpt':
            new_state_dict = dict()
            for k in dino_state_dict.keys():
                new_k = k.replace("mlp.", "")
                new_state_dict[new_k] = dino_state_dict[k]
            extra_tokens = dino_state_dict['pos_embed'][:, :1]
            src_weight = dino_state_dict['pos_embed'][:, 1:]
            src_weight = src_weight.reshape(1, 37, 37, 768).permute(0, 3, 1, 2)

            dst_weight = F.interpolate(
                src_weight.float(), size=16, align_corners=False, mode='bilinear')
            dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
            dst_weight = dst_weight.to(src_weight.dtype)
            new_state_dict['pos_embed'] = torch.cat((extra_tokens, dst_weight), dim=1)
            model = adaptformer.VisionTransformer(patch_size=14, tuning_config =  tuning_config, use_dinov2=True)
        if args.adapter == 'lora':
            model = rein.LoRADinoVisionTransformer(model)
            new_state_dict = dict()
            for k in dino_state_dict.keys():
                new_k = k.replace("attn.qkv", "attn.qkv.qkv")
                new_state_dict[new_k] = dino_state_dict[k]

            model.dino.load_state_dict(new_state_dict, strict=False)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)  

    elif args.net == 'dinov1':
        model_ = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        variant = dino_variant._dinov1_variant
        dino_state_dict = model_.state_dict()
        # print(dino_state_dict.keys())
        new_state_dict = dict()
        for k in dino_state_dict.keys():
            new_k = k.replace("mlp.", "")
            new_state_dict[new_k] = dino_state_dict[k]
        if args.adapter == 'rein':
            model = rein.ReinsDinoVisionTransformer(
                **variant
            )
        if args.adapter == 'adaptformer' or args.adapter == 'vpt':
            model = adaptformer.VisionTransformer(patch_size=16, tuning_config =  tuning_config)
        if args.adapter == 'lora':
            model_ = timm.create_model('vit_base_patch16_224', pretrained=False)
            model_.load_state_dict(dino_state_dict, strict=False)
            model = lora.LoRA_ViT_timm(model_, 4, 4)
        else:
            model.load_state_dict(new_state_dict, strict=True)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)  
        # print('aa')
    elif args.net == 'clip':
        # print(open_clip.list_pretrained())
        variant = dino_variant._clip_variant
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        # model, preprocess = clip.load("ViT-B/16", device=device)
        clip_state_dict = model.state_dict()

        state_dict = {}

        for k in clip_state_dict.keys():
            if k.startswith("visual."):
                new_k = k.replace("visual.transformer.res", "")
                new_k = new_k.replace('ln_', 'norm')
                new_k = new_k.replace('in_proj_', 'qkv.')
                new_k = new_k.replace('out_proj', 'proj')
                new_k = new_k.replace('c_fc', 'fc1')
                new_k = new_k.replace('c_proj', 'fc2')

                new_k = k.replace("visual.", "")
                new_k = k.replace("class_embedding", "cls_token")


                state_dict[new_k] = clip_state_dict[k]


        state_dict["pos_embed"] = clip_state_dict["positional_embedding"]

        # conv1 = clip_state_dict["conv1.weight"]
        # state_dict["conv1.weight"] = conv1

        model = rein.ReinsDinoVisionTransformer(
            **variant
        )
        u, w = model.load_state_dict(state_dict, True)
        print(u, w, "are misaligned params in vision transformer")

        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)  

    elif args.net == 'mae':
        variant = dino_variant._dinov1_variant
        dino_state_dict = torch.load('mae_pretrain_vit_base.pth')['model']
        new_state_dict = dict()
        for k in dino_state_dict.keys():
            new_k = k.replace("mlp.", "")
            new_state_dict[new_k] = dino_state_dict[k]
            
        if args.adapter == 'rein':
            model = rein.ReinsDinoVisionTransformer(
                **variant
            )
        if args.adapter == 'adaptformer' or args.adapter == 'vpt':
            model = adaptformer.VisionTransformer(patch_size=16, tuning_config =  tuning_config)
        # if args.adapter == 'lora':
        #     model_ = timm.create_model('vit_base_patch16_224', pretrained=False)
        #     model_.load_state_dict(dino_state_dict, strict=False)
        #     model = lora.LoRA_ViT_timm(model_, 4, 4)
        else:
            model.load_state_dict(new_state_dict, strict=True)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)  

    # elif args.net == 'bioCLIP':
    #     from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
    #     variant = dino_variant._dinov1_variant

    #     model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    #     tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    #     # print(model)
    #     model = model.visual.trunk
    #     bioCLIP_state_dict = model.state_dict()
    #     new_state_dict = dict()
    #     for k in bioCLIP_state_dict.keys():
    #         new_k = k.replace("mlp.", "")
    #         new_state_dict[new_k] = bioCLIP_state_dict[k]

    #     if args.adapter == 'rein':
    #         model = rein.ReinsDinoVisionTransformer(
    #             **variant
    #         )
    #     if args.adapter == 'adaptformer' or args.adapter == 'vpt':
    #         model = adaptformer.VisionTransformer(patch_size=16, tuning_config =  tuning_config)
            
    #     if args.adapter == 'lora':
    #         model_ = timm.create_model('vit_base_patch16_224', pretrained=False)
    #         model_.load_state_dict(bioCLIP_state_dict, strict=False)
    #         model = lora.LoRA_ViT_timm(model_, 4, 4)
    #     else:
    #         model.load_state_dict(new_state_dict, strict=True)
    #     model.linear = nn.Linear(768, config['num_classes'])
    #     model.to(device)  
        
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    
    if args.adapter == 'adaptformer' or args.adapter == 'vpt' or args.adapter == 'lora':
        set_requires_grad(model, ['adapt', 'linear', 'embeddings'])
        # print(params)
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay = 1e-05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)

    # if args.adapter == 'adaptformer' or args.adapter == 'vpt':
        # optimizer = torch.optim.SGD(model.linear.parameters(), lr=1e-1, momentum=0.9, weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, )
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 1) 
    print(train_loader.dataset[0][0].shape)

    # f = open(os.path.join(save_path, 'epoch_acc.txt'), 'w')
    avg_accuracy = 0.0
    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            features = model.forward_features(inputs)
            if args.adapter == 'rein':
                features = features[:, 0, :]
            # print(features.shape)
            outputs = model.linear(features)
            loss = criterion(outputs, targets)
            loss.backward()            
            optimizer.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')
            train_accuracy = correct/total
                  
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0

        valid_accuracy = utils.validation_accuracy_rein(model, valid_loader, device, args.adapter, False)
        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy 
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())

    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))
    
if __name__ =='__main__':
    train()