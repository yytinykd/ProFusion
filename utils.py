from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
from PIL import Image
import pickle
import numpy as np
from tqdm import tqdm
import random
import argparse
import yaml
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer
import pandas as pd
from openpyxl import load_workbook
from info_nce import InfoNCE



def get_seed():
    return 1
def dir_exists(path):
    return os.path.exists(path)

def seed_worker(worker_id):
    worker_seed = get_seed()
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save(obj, filepath, msg):
    """
    Saves the input object as a pickle file
    """
    print(f"Saving {msg} to {filepath}")
    with open(filepath, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(filepath, msg):
    print(f"Loading {msg} from {filepath}")
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config',
                        help='settings of ProFusion in yaml format', required=True)
    parser.add_argument('--alpha', dest='alpha',
                        help='alpha', required=False)
    parser.add_argument('--beta', dest='beta',help='beta',
                         required=False)
    parser.add_argument('--theta', dest='theta', help='theta',
                        type=float, required=False)
    parser.add_argument('--shots', dest='shots',
                        help='shots', type=int, required=False)
    parser.add_argument('--backbone', dest='backbone',
                        help='backbones: [base, large]', type=str, required=False)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset alias: [ caltech101, dtd, eurosat.sh, fgvc, food101, imagenet, oxford_flowers, oxford_pets, stanford_cars, sun397, ucf101 ]', required=False)
    args = parser.parse_args()
    return args


def update_cfg(cfg, args):
    if args.alpha:
        cfg['alpha'] = float(args.alpha)
    if args.beta:
        cfg['beta'] = float(args.beta)
    if args.theta:
        cfg['theta'] = args.theta
    if args.shots:
        cfg['shots'] = args.shots
    if args.backbone:
        cfg['backbone'] = args.backbone
    if args.dataset:
        cfg['dataset'] = args.dataset
    return cfg



# save token_list
def save_tokens_list(path, tokens_list):
    path = os.path.join(path, 'tokens_list.pkl')
    with open(path, 'wb') as f:
        pickle.dump(tokens_list, f)

# read token_list
def read_tokens_list(path):
    path = os.path.join(path, 'tokens_list.pkl')
    with open(path, 'rb') as f:
        tokens_list = pickle.load(f)
    return tokens_list

# save padding_mask_list
def save_padding_mask_list(path, padding_mask_list):
    path = os.path.join(path, 'padding_mask_list.pkl')
    with open(path, 'wb') as f:
        pickle.dump(padding_mask_list, f)

# read padding_mask_list
def read_padding_mask_list(path):
    path = os.path.join(path, 'padding_mask_list.pkl')
    with open(path, 'rb') as f:
        padding_mask_list = pickle.load(f)
    return padding_mask_list

# duplicate tokens_list padding_mask_list
def duplicate_elements(tokens_list, padding_mask_list, shots):
    duplicated_tokens_list = []
    duplicated_padding_mask_list = []

    for token in tokens_list:
        duplicated_tokens_list.extend([token] * shots)

    for mask in padding_mask_list:
        duplicated_padding_mask_list.extend([mask] * shots)

    return duplicated_tokens_list, duplicated_padding_mask_list

# text image fuse
def text_img_fuse(cfg, train_loader_fuse, tokens_list, padding_mask_list, beit3_model):

    fuse_feature_list = []
    tokens_list = np.array(tokens_list)
    padding_mask_list = np.array(padding_mask_list)
    tokens_list, padding_mask_list = torch.tensor(tokens_list), torch.tensor(padding_mask_list)

    model_dir_root = get_model_dir_root(cfg)
    path  = os.path.join(model_dir_root, 'fuse_feature_tensor.pth')


    if dir_exists(path):
        fuse_feature_tensor = torch.load(path)
        return fuse_feature_tensor
    else:
        with torch.no_grad():
            for i, ((data, target), token, padding_mask) in enumerate(tqdm(zip(train_loader_fuse, tokens_list, padding_mask_list),
                                                                           total=len(tokens_list))):

                token = token.unsqueeze(0)
                padding_mask = padding_mask.unsqueeze(0)
                image, token, padding_mask = data.cuda(), token.cuda(), padding_mask.cuda()
                vl_feature = beit3_model.encode_vl(image, token, padding_mask)
                fuse_feature_list.append(vl_feature)

            fuse_feature_tensor = torch.cat(fuse_feature_list, dim=0)
            fuse_feature_tensor = fuse_feature_tensor.view(-1, cfg['shots'], fuse_feature_tensor.shape[1])
            fuse_feature_tensor = fuse_feature_tensor.mean(dim=1)
            fuse_feature_tensor = F.normalize(fuse_feature_tensor, p=2, dim=1)
            fuse_feature_tensor = fuse_feature_tensor.t()

            # save
            torch.save(fuse_feature_tensor, path)
            return fuse_feature_tensor



def textual_features(cfg, classnames, template, _get_text_segment, beit3_model):
    msg = "text_memory_bank"
    model_dir_root = get_model_dir_root(cfg)
    os.makedirs(model_dir_root, exist_ok=True)
    beit_path = os.path.join(model_dir_root, f"text_mb_{beautify(cfg['backbone'])}_K_{cfg['shots']}.pkl")
    path = os.path.join(model_dir_root, "tokens_list.pkl")

    if dir_exists(path) and dir_exists(beit_path):
        text_prompts = classnames
        tokens_list = read_tokens_list(model_dir_root)
        padding_mask_list = read_padding_mask_list(model_dir_root)
        return load(beit_path, msg), tokens_list, padding_mask_list
    else:
        # Textual features
        text_prompts, textual_memory_bank, tokens_list, padding_mask_list = beit3_template_classifier(classnames, template,
                                                                                _get_text_segment, beit3_model)
        save(textual_memory_bank, beit_path, msg)
        save_tokens_list(model_dir_root, tokens_list)
        save_padding_mask_list(model_dir_root, padding_mask_list)
        return textual_memory_bank, tokens_list, padding_mask_list

def beit3_template_classifier(classnames, template, _get_text_segment, beit3_model):
    with torch.no_grad():
        beit_weights = []
        tokens_list, padding_mask_list = [], []
        for classname in tqdm(classnames):
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts, padding_mask, _ = _get_text_segment(texts[0])

            tokens_list.append(texts)
            padding_mask_list.append(padding_mask)

            texts, padding_mask = torch.tensor(texts).cuda(), torch.tensor(padding_mask).cuda()

            with torch.no_grad():
                class_embeddings = beit3_model.encode_text(texts.unsqueeze(0), padding_mask.unsqueeze(0))
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                beit_weights.append(class_embedding)

        beit_weights = torch.stack(beit_weights, dim=1).cuda()


    return classnames, beit_weights, tokens_list, padding_mask_list



def InfoNCELoss(A, B):

    loss = InfoNCE()
    return loss(A, B)




def compute_loss_and_matches(p, target_inds, z_img_proto, z_text_proto, cfg):
    pred_p, y_hat = p.max(dim=1)
    matches = (y_hat == target_inds).float().sum()
    loss = 0

    # cls loss
    nloss = nn.NLLLoss()
    loss += nloss(torch.log(p), target_inds)

    # img to text alignment loss
    img2txt_align_loss = InfoNCELoss(z_img_proto, z_text_proto)
    loss += img2txt_align_loss

    # text to img alignment loss
    txt2img_align_loss = InfoNCELoss(z_text_proto, z_img_proto)
    loss += txt2img_align_loss

    return matches, loss



def alpha_beta_theta_hp(alpha, beta, theta, val_accuracy, test_accuracy, cfg):

    best_val_acc, best_val_acc_idx = val_accuracy.max(), val_accuracy.argmax()
    best_test_acc, best_test_acc_idx = test_accuracy.max(), test_accuracy.argmax()

    best_val_alpha, best_val_beta, best_val_theta = alpha[best_val_acc_idx], beta[best_val_acc_idx], theta[best_val_acc_idx]
    best_test_alpha, best_test_beta, best_test_theta = alpha[best_test_acc_idx], beta[best_test_acc_idx], theta[best_test_acc_idx]

    print(f"alpha: {best_val_alpha: .3f}, b:{best_val_beta: .3f}, beta:{best_val_theta: .3f} | Max val-acc: {best_val_acc*100: .3f} | Max test-acc-using-val-alpha-beta: {test_accuracy[best_val_acc_idx]*100: .3f}")

    return best_val_alpha, best_val_beta, best_val_theta


def P_img_text_fusion(zq_imgs_flat, z_img_proto, z_text_proto, fuse_proto, alpha, beta, theta):

    gamma = abs(round((1-alpha-beta), 1))

    xq_img_proto_dists = torch.cdist(zq_imgs_flat.float(), z_img_proto.float(), p=2).pow(2)
    xq_text_proto_dists = torch.cdist( zq_imgs_flat.float(), z_text_proto.float(), p=2).pow(2)
    xq_fuse_proto_dists = torch.cdist( zq_imgs_flat.float(), fuse_proto.float(), p=2).pow(2)

    p_i = F.softmax(theta*(-xq_img_proto_dists), dim=1)
    p_t = F.softmax(theta*(-xq_text_proto_dists), dim=1)
    p_f = F.softmax(theta*(-xq_fuse_proto_dists), dim=1)

    p =  alpha * p_i + beta * p_t + gamma * p_f

    return p



def beautify(string):
    return string.strip().replace('/', '_').replace('-', '_')


def get_model_dir_root(cfg):
    return f"{cfg['cache_dir']}/models/{beautify(cfg['backbone'])}/K-{cfg['shots']}"


def build_cache_model(cfg, beit_model, train_loader_cache):

    model_dir_root = get_model_dir_root(cfg) + '/aug'
    os.makedirs(model_dir_root, exist_ok=True)
    def get_filename(cfg, type):
        return f"{model_dir_root}/visual_mb_{type}_aug_{cfg['augment_epoch']}_{cfg['shots']}_shots.pt"

    key_path = get_filename(cfg, 'keys')
    value_path = get_filename(cfg, 'values')

    if dir_exists(key_path) and dir_exists(value_path):
        cache_keys = torch.load(key_path)
        cache_values = torch.load(value_path)
    else:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))

                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    print(f"i: {i}, image.size(): {images.size()}, target.size(): {target.size()}" )
                    images = images.cuda()
                    image_features = beit_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)



        cache_values = torch.cat(cache_values, dim=0)
        index = torch.argsort(cache_values)
        cache_values = cache_values[index]
        cache_keys = cache_keys[:, index]
        cache_values = F.one_hot(cache_values)

        torch.save(cache_keys, key_path)
        torch.save(cache_values, value_path)


    return cache_keys, cache_values

def build_beit_cache_model(cfg, beit_model, train_loader_cache):

    model_dir_root = get_model_dir_root(cfg) + '/aug'
    os.makedirs(model_dir_root, exist_ok=True)
    def get_filename(cfg, type):
        return f"{model_dir_root}/visual_mb_{type}_aug_{cfg['augment_epoch']}_{cfg['shots']}_beit_shots.pt"

    key_path = get_filename(cfg, 'keys')
    value_path = get_filename(cfg, 'values')

    if dir_exists(key_path) and dir_exists(value_path):
        cache_keys = torch.load(key_path)
        cache_values = torch.load(value_path)
    else:
        cache_keys = []
        cache_values = []

        with torch.no_grad():

            for augment_idx in range(cfg['augment_epoch']):
                train_features = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))

                for i, (images, target) in enumerate(tqdm(train_loader_cache)):

                    images = images.cuda()
                    image_features = beit_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)



        cache_values = torch.cat(cache_values, dim=0)
        index = torch.argsort(cache_values)
        cache_values = cache_values[index]
        cache_keys = cache_keys[:, index]
        cache_values = F.one_hot(cache_values)

        torch.save(cache_keys, key_path)
        torch.save(cache_values, value_path)


    return cache_keys, cache_values


def build_visual(cfg, beit3_model, train_loader_cache):

    model_dir_root = get_model_dir_root(cfg) + '/aug'
    os.makedirs(model_dir_root, exist_ok=True)
    def get_filename(cfg, type):
        return f"{model_dir_root}/visual_mb_{type}_aug_{cfg['augment_epoch']}_{cfg['shots']}_shots.pt"
    key_path = get_filename(cfg, 'keys')
    value_path = get_filename(cfg, 'values')

    if dir_exists(key_path) and dir_exists(value_path):
        cache_keys = torch.load(key_path, weights_only=True)
        cache_values = torch.load(value_path, weights_only=True)
    else:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = beit3_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)

        cache_values = torch.cat(cache_values, dim=0)
        index = torch.argsort(cache_values)
        cache_values = cache_values[index]
        cache_keys = cache_keys[:, index]
        cache_values = F.one_hot(cache_values)

        torch.save(cache_keys, key_path)
        torch.save(cache_values, value_path)

    return cache_keys, cache_values


def load_features(cfg, split, beit3_model, loader):
    root_dir_prefix = f"{get_model_dir_root(cfg)}/{split}"
    feature_path = f"{root_dir_prefix}_features.pt"
    label_path = f"{root_dir_prefix}_labels.pt"

    if dir_exists(feature_path) and dir_exists(label_path):
        print(f"Loading cached features and labels from {root_dir_prefix}")
        features = torch.load(feature_path, weights_only=True)
        labels = torch.load(label_path, weights_only=True)
    else:
        print(f"Creating cached (features, labels) and saving to {root_dir_prefix}")
        features, labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = beit3_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)
        torch.save(features, feature_path)
        torch.save(labels, label_path)

    return features, labels

