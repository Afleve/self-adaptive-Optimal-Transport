import os
import random
import argparse
import numpy as np
from datasets import get_all_dataloaders
from utils import *

import warnings

warnings.filterwarnings("ignore")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='caltech101', help='dataset name', type=str)
    parser.add_argument('--root_path', default='/data/Public', type=str)
    parser.add_argument('--method', default='OptimalTrans', type=str, help='')
    parser.add_argument('--shots', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--backbone', default='vit_b16', type=str)
    parser.add_argument('--visionLanguageM', default='clip', type=str)
    parser.add_argument('--visionFMs', default=['dinov3', 'dinov2'], type=list)

    parser.add_argument('--prototypes_path', default="./prototypes")
    parser.add_argument('--setting', default="")
    parser.add_argument('--prototypes_method', default="")
    parser.add_argument('--prototypes_dataset', default="")
    parser.add_argument('--prototypes_shots', default=16)

    parser.add_argument('--root', default='temp', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--cat_bool', action='store_true')

    args = parser.parse_args()

    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    backbones = {'rn50': 'RN50',
                 'rn101': 'RN101',
                 'vit_b16': 'ViT-B/16',
                 'vit_b32': 'ViT-B/32',
                 'vit_l14': 'ViT-L/14'}

    # Load config file
    args = get_arguments()

    cfg = {'dataset': args.dataset}
    cache_dir = os.path.join(f'./caches/{args.backbone}', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    cfg['load_cache'] = True
    cfg['load_pre_feat'] = True


    cfg['backbone'] = backbones[args.backbone]

    cfg['seed'] = args.seed
    cfg['root_path'] = args.root_path
    cfg['method'] = args.method
    cfg['shots'] = args.shots

    cfg['prototypes_path'] = args.prototypes_path
    cfg['setting'] = args.setting
    cfg['prototypes_method'] = args.prototypes_method
    cfg['prototypes_dataset'] = args.prototypes_dataset
    cfg['prototypes_shots'] = args.prototypes_shots

    cfg['root'] = args.root
    cfg['cat_bool'] = args.cat_bool
    cfg['device'] = args.device
    cfg['backbone_name'] = args.backbone

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    set_random_seed(args.seed)

    train_loader, val_loader, test_loader, dataset = get_all_dataloaders(cfg, preprocess)

    _, _, _, _, vlm_feature, true_labels, clip_prototypes = get_all_features(
        cfg, train_loader, val_loader, test_loader, dataset, clip_model, args.visionLanguageM)
    clip_model = clip_model.to('cpu')  # unload CLIP model from VRAM

    vlm_features = [vlm_feature.float()]
    # visionFM
    vfm_features = []
    for visionFM in args.visionFMs:
        if visionFM == 'dinov2':
            import models.dinov2.visiontransformer as vits
            pretrain_path = 'yours/path/to/dinov2_vitb16_pretrain_lvd1689m-73cec8be.pth'
            dinov2_model = vits.__dict__['vit_base']().cuda()
            torch.cuda.empty_cache()
            state_dict = torch.load(pretrain_path, map_location='cuda')
            dinov2_model.load_state_dict(state_dict)
            dinov2_model.eval()
            vfm_feature, _ = pre_load_features(cfg, "test", dinov2_model, test_loader, visionLanguageM='dinov2')
        if visionFM == "dinov3":
            url = "yours/.cache/torch/hub/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            dinov3_model = torch.hub.load('yours/dinov3/', 'dinov3_vitb16', source='local', weights=url).cuda()
            dinov3_model.eval()
            vfm_feature, _ = pre_load_features(cfg, "test", dinov3_model, test_loader, visionLanguageM='dinov3')

        vfm_features.append(vfm_feature.cuda())

    if args.method == 'OptimalTrans':
        from OptimalTrans_solver.OptimalTrans_utils import prepare_objects
        y_hat = prepare_objects(vlm_feature, true_labels, clip_prototypes)
        from OptimalTrans_solver.OptimalTrans_auto import OptimalTrans
        OptimalTrans(vlm_features, vfm_features, true_labels, [y_hat], cfg)
    # elif args.method == 'TransCLIP':
    #     from TransCLIP_solver.TransCLIP import TransCLIP_solver
    #     if cfg['shots'] == 0:
    #         TransCLIP_solver(shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
    #                          clip_prototypes)
    #     else:
    #         shot_features = shot_features.unsqueeze(0)
    #         TransCLIP_solver(shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
    #                          clip_prototypes)

    # elif args.method == 'On-top':

    #     from TransCLIP_solver.TransCLIP import TransCLIP_solver
    #     from TransCLIP_solver.bench_utils import prepare_for_bench

    #     test_features, test_labels, initial_prototypes, initial_logits, clip_prototypes = prepare_for_bench(args,
    #                                                                                                         test_features,
    #                                                                                                         test_labels,
    #                                                                                                         clip_prototypes)

    #     if cfg['shots'] == 0:
    #         TransCLIP_solver(shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
    #                          clip_prototypes, initial_prototypes, initial_logits)
    #     else:
    #         shot_features = shot_features.unsqueeze(0)
    #         TransCLIP_solver(shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
    #                          clip_prototypes, initial_prototypes, initial_logits)


    # elif args.method == 'transductive_finetuning':
    #     device = clip_prototypes.device
    #     shot_features = shot_features.type(torch.FloatTensor).squeeze().to(device)
    #     val_features = val_features.type(torch.FloatTensor).squeeze().to(device)
    #     test_features = test_features.type(torch.FloatTensor).squeeze().to(device)
    #     clip_prototypes = clip_prototypes.type(torch.FloatTensor).squeeze().to(device)
    #     from baselines.runner import run_transductive_finetuning
    #     run_transductive_finetuning(cfg, shot_features, shot_labels, val_features, val_labels, test_features,
    #                                 test_labels,
    #                                 clip_prototypes)
    # elif args.method == 'bdcspn':
    #     device = clip_prototypes.device
    #     shot_features = shot_features.type(torch.FloatTensor).squeeze().to(device)
    #     val_features = val_features.type(torch.FloatTensor).squeeze().to(device)
    #     test_features = test_features.type(torch.FloatTensor).squeeze().to(device)
    #     clip_prototypes = clip_prototypes.type(torch.FloatTensor).squeeze().to(device)
    #     from baselines.runner import run_bdcspn
    #     run_bdcspn(cfg, shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
    #                clip_prototypes)
    # elif args.method == 'laplacian_shot':
    #     device = clip_prototypes.device
    #     shot_features = shot_features.type(torch.FloatTensor).squeeze().to(device)
    #     val_features = val_features.type(torch.FloatTensor).squeeze().to(device)
    #     test_features = test_features.type(torch.FloatTensor).squeeze().to(device)
    #     clip_prototypes = clip_prototypes.type(torch.FloatTensor).squeeze().to(device)
    #     from baselines.runner import run_laplacian_shot
    #     run_laplacian_shot(cfg, shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
    #                        clip_prototypes)
    # elif args.method == 'ptmap':
    #     device = clip_prototypes.device
    #     shot_features = shot_features.type(torch.FloatTensor).squeeze().to(device)
    #     val_features = val_features.type(torch.FloatTensor).squeeze().to(device)
    #     test_features = test_features.type(torch.FloatTensor).squeeze().to(device)
    #     clip_prototypes = clip_prototypes.type(torch.FloatTensor).squeeze().to(device)
    #     from baselines.runner import run_pt_map
    #     run_pt_map(cfg, shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
    #                clip_prototypes)
    # elif args.method == 'tim':
    #     device = clip_prototypes.device
    #     shot_features = shot_features.type(torch.FloatTensor).squeeze().to(device)
    #     val_features = val_features.type(torch.FloatTensor).squeeze().to(device)
    #     test_features = test_features.type(torch.FloatTensor).squeeze().to(device)
    #     clip_prototypes = clip_prototypes.type(torch.FloatTensor).squeeze().to(device)
    #     from baselines.runner import run_tim
    #     run_tim(cfg, shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
    #             clip_prototypes, version='adm')


if __name__ == '__main__':
    main()
