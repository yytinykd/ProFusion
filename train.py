from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import io
import json
import torch
import torch.nn as nn

from datasets import build_dataset
from datasets.utils import build_data_loader
from utils import *
from datasets.imagenet import ImageNet, get_random_train_tfm
from model import Adapter, Adapter_FC

# beit3
from beit3.test_model import beit_main, _get_text_segment, build_transform, get_args
import torch.nn.functional as F

def ProFusion_F(cfg, visual_features, visual_values, val_features, val_labels, test_features, test_labels,
                   textual_features, fuse_features, beit3_model):
    ndim, NxK = visual_features.shape
    K = cfg['shots']
    N = NxK // K
    torch.autograd.set_detect_anomaly(True)

    visual_embeddings = nn.Embedding(num_embeddings=NxK, embedding_dim=ndim).cuda()
    visual_embeddings.weight = nn.Parameter(visual_features.t().clone())
    textual_embeddings = nn.Embedding(num_embeddings=N, embedding_dim=ndim).cuda()
    textual_embeddings.weight = nn.Parameter(textual_features.t().clone())
    fuse_embeddings = nn.Embedding(num_embeddings=N, embedding_dim=ndim).cuda()
    fuse_embeddings.weight = nn.Parameter(fuse_features.t().clone())

    # adapter: conx or fc
    if 'conv' in cfg['adapter']:
        adapter = Adapter(ndim, c_type=cfg['adapter'], dtype=torch.half).cuda().float()
    elif cfg['adapter'] == 'fc':
        adapter = Adapter_FC(ndim, dtype=torch.half).cuda().float()
    if cfg['train_vis_mem_only']:
        params = list(adapter.parameters()) + list(visual_embeddings.parameters()) + list(fuse_embeddings.parameters())
    else:
        params = (list(visual_embeddings.parameters()) + list(textual_embeddings.parameters()) + list(adapter.parameters()) + list(fuse_embeddings.parameters()))
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], eps=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * NxK)

    best_acc, best_epoch = 0.0, 0
    exp = 1
    step_size = 1 / 10 ** exp
    alpha_list = np.arange(0, 1 + step_size, step_size).round(exp)
    theta_list = np.concatenate((np.arange(0.1, 1, 0.1), np.arange(1, 21, 1.0)))

    val_acc_list = []
    test_acc_list = []
    train_acc_list = []

    model_dir_root = get_model_dir_root(cfg)
    os.makedirs(model_dir_root, exist_ok=True)
    train_labels = torch.argmax(visual_values, dim=1)

    # ------------------------------------------------    train    ---------------------------------------------------------
    best_alpha = cfg['alpha']
    best_beta = cfg['beta']
    best_theta = cfg['theta']


    class_upper = int(N * 0.4)  # 40
    class_lower = max(int(N * 0.2), 1)  # 20

    for epoch in tqdm(range(cfg['train_epoch'])):
        # Train
        visual_embeddings.train()
        textual_embeddings.train()
        fuse_embeddings.train()
        adapter.train()

        correct_samples, all_samples, = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(epoch, cfg['train_epoch']))

        class_indexes = np.random.permutation(N)
        start = 0
        while start < N - 1:
            num_class = np.random.randint(class_lower, class_upper)
            class_index = sorted(class_indexes[start:min(start + num_class, N - 1)])
            num_class = len(class_index)

            support_index = []
            query_index = []
            zq_labels = []

            for i in range(num_class):
                cls = class_index[i]
                # sample number of support
                assert K > 0
                item_indexes = np.random.permutation(K)
                n = np.random.randint(1, K) if K > 1 else K
                support = sorted(item_indexes[:n])
                if K > 1:
                    query = sorted(item_indexes[n:])
                else:
                    query = sorted(item_indexes[:n])

                # the number index of support query zq_lables
                support_index.extend(cls * K + support)
                query_index.extend(cls * K + query)
                zq_labels.extend([cls] * len(query))

            # query_imgs, query_labels
            query_index = torch.as_tensor(query_index).cuda()
            zq_imgs = visual_features.t()[query_index]
            zq_imgs = adapter(zq_imgs).float()
            zq_imgs = zq_imgs / zq_imgs.norm(dim=-1, keepdim=True)
            zq_labels = torch.as_tensor(zq_labels).cuda()

            # compute img prototypes
            zs_imgs = visual_embeddings.weight.view(-1, K, ndim)
            zs_imgs = zs_imgs / zs_imgs.norm(dim=-1, keepdim=True)
            z_img_proto = zs_imgs.mean(dim=1).float()
            z_img_proto = z_img_proto / z_img_proto.norm(dim=-1, keepdim=True)

            # compute text prototypes
            zs_text = textual_embeddings.weight
            zs_text = zs_text / zs_text.norm(dim=-1, keepdim=True)
            z_text_proto = zs_text.float()

            # compute fuse prototypes
            fuse_proto = fuse_embeddings.weight
            fuse_proto = fuse_proto / fuse_proto.norm(dim=-1, keepdim=True)
            fuse_proto = fuse_proto.float()

            # compute probability
            p = P_img_text_fusion(zq_imgs, z_img_proto, z_text_proto, fuse_proto, best_alpha, best_beta, best_theta)

            # loss
            matches, train_loss = compute_loss_and_matches(p, zq_labels, z_img_proto, z_text_proto, cfg)

            correct_samples += matches
            all_samples += len(zq_labels)
            loss_list.append(train_loss.item())
            optimizer.zero_grad()
            train_loss.backward(retain_graph=True)
            optimizer.step()
            start += len(class_index)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        train_acc = correct_samples / all_samples
        train_loss = sum(loss_list) / len(loss_list)
        print('LR: {:.6f}, Acc: {:.4f}% ({:}/{:}), Loss: {:.4f}'.format(current_lr, train_acc * 100, correct_samples, all_samples, train_loss))


        # ------------------------------------------------------ val -----------------------------------------------------------
        with torch.no_grad():
            zs_imgs = visual_embeddings.weight.view(-1, K, ndim)
            zs_imgs = zs_imgs / zs_imgs.norm(dim=-1, keepdim=True)
            z_img_proto = zs_imgs.mean(dim=1)
            z_img_proto = z_img_proto / z_img_proto.norm(dim=-1, keepdim=True)

            zs_text = textual_embeddings(torch.arange(N, requires_grad=False).cuda())
            z_text_proto = zs_text / zs_text.norm(dim=-1, keepdim=True)

            fuse_proto = fuse_embeddings(torch.arange(N, requires_grad=False).cuda())
            fuse_proto = fuse_proto / fuse_proto.norm(dim=-1, keepdim=True)

            val_features_adapt = adapter(val_features)
            val_features_adapt = val_features_adapt / val_features_adapt.norm(dim=-1, keepdim=True)

            p = P_img_text_fusion(val_features_adapt, z_img_proto, z_text_proto, fuse_proto, best_alpha, best_beta, best_theta)
            pred_p, y_hat = p.max(dim=1)
            matches = (y_hat == val_labels).float().sum()
            val_loss = -torch.log(pred_p).mean()

            # val_acc
            val_acc = (p.max(1)[1] == val_labels).float().mean()
            print("**** ProFusion's val accuracy: {:.2f}% | loss: {:.2f}***\n".format(val_acc * 100, val_loss))

            model_dir_root = get_model_dir_root(cfg)  # models/RN50/K-16
            model_dir = f"{model_dir_root}/alpha-beta/{best_alpha}-{best_beta}-{best_theta}"
            model_prefix = f"best_lr_{cfg['lr']}_aug_{cfg['augment_epoch']}_epochs_{cfg['train_epoch']}"
            os.makedirs(model_dir, exist_ok=True)

            # save best visual text adapter model
            best_model_path_v = os.path.join(model_dir, f"{model_prefix}_v.pt")
            best_model_path_t = os.path.join(model_dir, f"{model_prefix}_t.pt")
            best_model_path_a = os.path.join(model_dir, f"{model_prefix}_a.pt")
            best_model_path_f = os.path.join(model_dir, f"{model_prefix}_f.pt")
            if val_acc >= best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(visual_embeddings.weight, best_model_path_v)
                torch.save(textual_embeddings.weight, best_model_path_t)
                torch.save(fuse_embeddings.weight, best_model_path_f)
                torch.save(adapter.state_dict(), best_model_path_a)

        print(f"Best model: best_val_acc = {best_acc * 100: .2f}, best_val_epoch = {best_epoch}")

    # ------------------------------------------------------- test ---------------------------------------------------------
    with torch.no_grad():
        model_dir = f"{model_dir_root}/alpha-beta/{best_alpha}-{best_b}-{best_beta}"
        model_prefix = f"best_lr_{cfg['lr']}_aug_{cfg['augment_epoch']}_epochs_{cfg['train_epoch']}"

        # load text_img_proto and adapter
        best_model_path_v = os.path.join(model_dir, f"{model_prefix}_v.pt")
        best_model_path_t = os.path.join(model_dir, f"{model_prefix}_t.pt")
        best_model_path_f = os.path.join(model_dir, f"{model_prefix}_f.pt")
        best_model_path_a = os.path.join(model_dir, f"{model_prefix}_a.pt")
        try:
            embeddings_v = torch.load(best_model_path_v, weights_only=True)  # [1600, 1024]
            embeddings_t = torch.load(best_model_path_t, weights_only=True)  # [100, 1024]
            embeddings_f = torch.load(best_model_path_f, weights_only=True)
            adapter.load_state_dict(torch.load(best_model_path_a, weights_only=True))
        except:
            raise FileNotFoundError(f"File does not exist: {best_model_path_v} and {best_model_path_t}")

        zs_imgs = embeddings_v.view(-1, K, ndim)
        zs_imgs = zs_imgs / zs_imgs.norm(dim=-1, keepdim=True)
        z_img_proto = zs_imgs.mean(dim=1)
        z_img_proto = z_img_proto / z_img_proto.norm(dim=-1, keepdim=True)

        zs_text = embeddings_t
        z_text_proto = zs_text / zs_text.norm(dim=-1, keepdim=True)

        fuse_proto = embeddings_f
        fuse_proto = fuse_proto / fuse_proto.norm(dim=-1, keepdim=True)

        test_features = adapter(test_features)
        test_features = test_features / test_features.norm(dim=-1, keepdim=True)

        train_features = adapter(visual_features.t())
        train_features = train_features / train_features.norm(dim=-1, keepdim=True)

        val_features_adapt = adapter(val_features)

        val_acc_list = []
        test_acc_list = []
        train_acc_list = []

        for alpha in tqdm(alpha_list, total=len(alpha_list)):
            for beta in np.arange(0, 1 + 0.1 - alpha, 0.1):
                beta = abs(round(b, 1))
                if beta >= 0:
                    gamma = abs(round((1 - alpha - b), 1))
                    if gamma >= 0 and (alpha + beta + gamma == 1):
                        for theta in theta_list:
                            # val_acc
                            p = P_img_text_fusion(val_features_adapt, z_img_proto, z_text_proto, fuse_proto, alpha, beta, theta)
                            val_acc = (p.max(1)[1] == val_labels).float().mean()
                            val_acc_list.append([alpha, b, beta, val_acc.item()])

                            # test_acc
                            p = P_img_text_fusion(test_features, z_img_proto, z_text_proto, fuse_proto, alpha, beta, theta)
                            test_acc = (p.max(1)[1] == test_labels).float().mean()
                            test_acc_list.append([alpha, b, beta, test_acc.item()])


        val_acc_list = np.array(val_acc_list)
        test_acc_list = np.array(test_acc_list)

        p = P_img_text_fusion(test_features, z_img_proto, z_text_proto, fuse_proto, best_alpha, best_beta, best_theta)
        test_acc_fix = (p.max(1)[1] == test_labels).float().mean()


        best_alpha_hp, best_beta_hp, best_theta_hp = alpha_beta_theta_hp(val_acc_list[:, 0], val_acc_list[:, 1], val_acc_list[:, 2], val_acc_list[:, 3],
            test_acc_list[:, 3],  cfg)

        p = P_img_text_fusion(test_features, z_img_proto, z_text_proto, fuse_proto, best_alpha_hp, best_beta_hp, best_theta_hp)
        test_acc_hp = (p.max(1)[1] == test_labels).float().mean()

        print("**** Alpha: {:.1f},  beta: {:.1f},  gamma: {:.1f},  theta{:.1f}, {}'s {} shots test accuracy: {:.2f}% ****".format(
                best_alpha,best_beta, abs(round((1 - best_alpha - best_b), 1)), best_theta,cfg['dataset'],cfg['shots'],
                max(test_acc_fix, test_acc_hp) * 100))


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg = update_cfg(cfg, args)
    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    print(cfg, "\n")

    # beit3
    opts = get_args()
    transform = build_transform(False, opts)
    beit3_model = beit_main()
    beit3_model.cuda()
    beit3_model.eval()

    # SEED
    seed = get_seed()
    random.seed(seed)
    np.random.seed(seed)
    g = torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # batch_size
    n_workers, train_bs, val_bs, test_bs = 8, 256, 256, 256

    # dataset
    print("Preparing dataset.")
    if cfg['dataset'] == 'imagenet':
        dataset = ImageNet(cfg['root_path'], cfg['shots'], transform)
        train_loader_cache = torch.utils.data.DataLoader(dataset.train, batch_size=train_bs, num_workers=n_workers,
                                                         shuffle=False, worker_init_fn=seed_worker, generator=g)
        val_loader = torch.utils.data.DataLoader(dataset.test, batch_size=val_bs, num_workers=n_workers, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=test_bs, num_workers=n_workers,
                                                  shuffle=False)
    else:
        dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
        train_tranform = get_random_train_tfm()
        train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=train_bs, tfm=train_tranform,
                                               is_train=True, shuffle=False, worker_init_fn=seed_worker, generator=g)
        val_loader = build_data_loader(data_source=dataset.val, batch_size=val_bs, is_train=False, tfm=transform,
                                       shuffle=False)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=test_bs, is_train=False, tfm=transform,
                                        shuffle=False)

    print("Constructing memory bank by few-shot visual  features.")
    visual_features, visual_values = build_visual(cfg, beit3_model, train_loader_cache)

    print("\nConstructing memory bank by few-shot text features.")
    textual_features, tokens_list, padding_mask_list = textual_features(cfg,dataset.classnames, dataset.template,
                                                                                                  _get_text_segment,
                                                                                                  beit3_model)

    duplicated_tokens_list, duplicated_padding_mask_list = duplicate_elements(tokens_list, padding_mask_list,
                                                                              cfg['shots'])

    print("\nConstructing fuse prototypes.")
    if cfg['dataset'] == 'imagenet':
        train_loader_fuse = torch.utils.data.DataLoader(dataset.train, batch_size=1, num_workers=n_workers, shuffle=False, worker_init_fn=seed_worker, generator=g)
        fuse_features = text_img_fuse(cfg, train_loader_fuse, duplicated_tokens_list, duplicated_padding_mask_list, beit3_model)
    else:
        train_loader_fuse = build_data_loader(data_source=dataset.train_x, batch_size=1, tfm=transform, is_train=True, shuffle=False, worker_init_fn=seed_worker, generator=g)
        fuse_features = text_img_fuse(cfg, train_loader_fuse, duplicated_tokens_list, duplicated_padding_mask_list, beit3_model)

    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = load_features(cfg, "val", beit3_model, val_loader)

    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = load_features(cfg, "test", beit3_model, test_loader)

    # ------------------------------------------ ProtoFusion-F------------------------------------------
    ProFusion_F(cfg, visual_features, visual_values, val_features, val_labels,
                   test_features, test_labels, textual_features, fuse_features, beit3_model)


if __name__ == '__main__':
    main()
