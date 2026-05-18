import os
import random
import argparse
import numpy as np
from datasets import *
from utils import *
from OptimalTrans_solver.OptimalTrans_utils import prepare_objects
from OptimalTrans_solver.OptimalTrans_auto import OptimalTrans, evalute_test
from modelscope import AutoModel
import warnings
import time
import torch

warnings.filterwarnings("ignore")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imagenet', help='dataset name', type=str)
    parser.add_argument('--root_path', default='/data/Public', type=str)
    parser.add_argument('--method', default='SelfApativeOptimalTrans', type=str,
                        help='')
    parser.add_argument('--unsup_shots', default=-1, type=int)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--backbone', default='vit_b16', type=str)
    parser.add_argument('--vlms', default='clip', type=str)
    parser.add_argument('--vfms', default='dinov2,dinov3', type=str)
    parser.add_argument('--device', default='cuda', type=str)
        
    parser.add_argument('--eps', default=0.01, type=float)

    parser.add_argument('--prototypes_path', default="./prototypes")
    parser.add_argument('--setting', default="Transductive", choices=['Inductive', 'Transductive'])
    parser.add_argument('--prototypes_method', default="")
    parser.add_argument('--prototypes_dataset', default="")
    parser.add_argument('--prototypes_shots', default=16)

    args = parser.parse_args()

    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    # Load config file
    args = get_arguments()

    cfg = {'dataset': args.dataset}

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_root'] = cache_dir
    cfg['root'] = "records"

    signal = True
    cfg['load_cache'] = signal
    cfg['load_pre_feat'] = signal

    print("\nRunning configs.")


    cfg['seed'] = args.seed
    cfg['root_path'] = args.root_path
    cfg['method'] = args.method
    cfg['shots'] = args.unsup_shots

    cfg['prototypes_path'] = args.prototypes_path
    cfg['setting'] = args.setting
    cfg['prototypes_method'] = args.prototypes_method
    cfg['prototypes_dataset'] = args.prototypes_dataset
    cfg['prototypes_shots'] = args.prototypes_shots

    cfg['vlms'] = args.vlms.split(',')
    cfg['backbone'] = args.backbone.split(',')
    cfg['vfms'] = args.vfms.split(',')
    cfg['device'] = args.device

    print(cfg, "\n")

    # Prepare dataset
    set_random_seed(args.seed)

    print("Preparing dataset.")

    dataset = get_all_datasets(cfg, preprocess=None)

    print("Loading features.")

    vlms_test_features, vlms_train_features, vlms_test_probs, vlms_train_probs = [], [], [], []
    for vlm in cfg['vlms']:
        if vlm == 'clip':
            for backbone in cfg['backbone']:
                cfg['cache_dir'] = os.path.join(cfg['cache_root'], vlm, backbone)
                os.makedirs(cfg['cache_dir'], exist_ok=True)

                clip_model, preprocess = clip.load(backbones[backbone])
                clip_model.eval()
                train_loader, test_loader = dataset_2_dataloader(cfg, dataset, preprocess)
                
                shot_features, shot_labels, val_features, val_labels, test_features, test_labels, clip_prototypes = get_all_features(
                    cfg, train_loader, None, test_loader, dataset, clip_model)
                clip_model = clip_model.to('cpu')

                vlms_test_features.append(test_features)
                test_probs = prepare_objects(test_features, test_labels, clip_prototypes)
                vlms_test_probs.append(test_probs)

                torch.save(clip_prototypes, os.path.join(cfg['cache_dir'], 'clip_prototypes.pth'))

                if cfg['setting'] == 'Inductive':
                    vlms_train_features.append(shot_features)
                    train_probs = prepare_objects(shot_features, shot_labels, clip_prototypes)
                    vlms_train_probs.append(train_probs)

    vfms_test_features, vfms_train_features = [], []
    for vfm in cfg['vfms']:
        vfm_model = None
        if vfm == '':
            break
        if vfm == 'dinov2':
            if not cfg['load_cache'] or not cfg['load_pre_feat']:
                vfm_model = AutoModel.from_pretrained('facebook/dinov2-with-registers-large')
            preprocess = transform_v2
        if vfm == "dinov3":
            if not cfg['load_cache'] or not cfg['load_pre_feat']:
                vfm_model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
            preprocess = transform_v3
        cfg['cache_dir'] = os.path.join(cfg['cache_root'], vfm)
        os.makedirs(cfg['cache_dir'], exist_ok=True)
        if not cfg['load_cache'] or not cfg['load_pre_feat']:
            vfm_model = vfm_model.to(cfg['device'])
            vfm_model.eval()
        train_loader, test_loader = dataset_2_dataloader(cfg, dataset, preprocess)
        test_features, _ = pre_load_features(cfg, "test", vfm_model, test_loader, modelName=vfm)
        vfms_test_features.append(test_features)
        if cfg['setting']=="Inductive":
                train_features, _ = build_cache_model(cfg, vfm_model, train_loader, n_views=1, modelName=vfm)
                vfms_train_features.append(train_features.to(cfg['device']))

    if cfg['setting'] == 'Transductive':
        start_time = time.time()
        z, y_hats = OptimalTrans([*vlms_test_features, *vfms_test_features], test_labels, vlms_test_probs, cfg)
        print("Cost time:", time.time() - start_time)
    else:
        gmm_adapter_list, alpha, beta = OptimalTrans([*vlms_train_features, *vfms_train_features], shot_labels, vlms_train_probs, cfg)
        evalute_test(cfg, [*vlms_test_features, *vfms_test_features], test_labels, vlms_test_probs, gmm_adapter_list, alpha, beta)



if __name__ == '__main__':
    main()
