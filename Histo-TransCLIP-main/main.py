import os
import time
import random
import argparse
import clip
import numpy as np
from datasets_name import get_all_dataloaders
from utils import *


from models.plip.plip import PLIP
from models.CONCH.conch.open_clip_custom import create_model_from_pretrained
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch
from models.MUSK.musk import utils, modeling
from timm.models import create_model
from datasets_name.utils import *
from models.MUSK.musk_trans import musk_transform

from warnings import filterwarnings
filterwarnings("ignore")
import sys
sys.path.append("../../SOTA")
from OptimalTrans_solver.OptimalTrans_auto import *



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='nct', help="dataset name - sicap_mil - skincancer - lc_lung - nct - pannuke - WSSS4LUAD", type=str)
    parser.add_argument('--root_path', default='/data/datasets', type=str)
    parser.add_argument('--method', default='TransCLIP', type=str,
                        help='')
    parser.add_argument('--model', default='PLIP', type=str, help='')
    parser.add_argument('--selected_model', default='CLIP,CONCH', type=str, help='')

    parser.add_argument('--shots', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--backbone', default='vit_b16', type=str)

    parser.add_argument('--prototypes_path', default="./prototypes")
    parser.add_argument('--setting', default="")
    parser.add_argument('--prototypes_method', default="")
    parser.add_argument('--prototypes_dataset', default="")
    parser.add_argument('--prototypes_shots', default=16)
    
    parser.add_argument('--root', default='test')

    args = parser.parse_args()

    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_objects(query_features, query_labels, clip_prototypes):

    query_features = query_features.cuda().float()
    query_labels = query_labels.cuda()
    clip_prototypes = clip_prototypes.cuda().float()

    # if len(clip_prototypes.shape) == 3:  # use more than 1 template
    #     clip_prototypes = clip_prototypes[0]  # use only the first one
    clip_prototypes = clip_prototypes.mean(dim=0) # clip_prototypes[0] use only the first one clip_prototypes.mean(dim=0)
    clip_prototypes /= clip_prototypes.norm(dim=0, keepdim=True)

    clip_logits = 100 * query_features @ clip_prototypes

    return clip_logits

def main():
    backbones = {'rn50': 'RN50',
                 'rn101': 'RN101',
                 'vit_b16': 'ViT-B/16',
                 'vit_b32': 'ViT-B/32',
                 'vit_l14': 'ViT-L/14'}

    # Load config file
    args = get_arguments()

    cfg = {'dataset': args.dataset}

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    cfg['load_cache'] = True
    cfg['load_pre_feat'] = True

    print("\nRunning configs.")

    cfg['backbone'] = backbones[args.backbone]

    cfg['seed'] = args.seed
    cfg['root_path'] = args.root_path
    cfg['method'] = args.method
    cfg['model'] = args.model
    cfg['selected_model'] = args.selected_model
    cfg['shots'] = args.shots

    cfg['prototypes_path'] = args.prototypes_path
    cfg['setting'] = args.setting
    cfg['prototypes_method'] = args.prototypes_method
    cfg['prototypes_dataset'] = args.prototypes_dataset
    cfg['prototypes_shots'] = args.prototypes_shots

    # print(cfg, "\n")

    cfg['file'] = f'./res/{args.root}'
    os.makedirs(cfg['file'], exist_ok=True)
    
    # Prepare dataset
    set_random_seed(args.seed)

    # Clip
    clip_model, preprocess_val = clip.load('ViT-B/16')
    # CONCH
    conch_model, conch_preprocess_val = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="models/CONCH/conch/pytorch_model.bin")
    # PLIP
    local_path="models/vinid/plip"
    plip = PLIP(local_path)
    plip_model = plip.model
    preprocess = plip.preprocess
    _, plip_preprocess_val = clip.load("ViT-B/16")

    #MUSK
    musk_model = create_model("musk_large_patch16_384")
    utils.load_model_and_may_interpolate(
        ckpt_path="models/MUSK/musk/models/model.safetensors",  
        model = musk_model,            
        model_key='model|module',    
        model_prefix=''      
    )
    musk_preprocess_val = musk_transform()

    models = {
        'CLIP' : clip_model,
        'CONCH' : conch_model,
        'PLIP' : plip_model,
        'MUSK':musk_model
    }

    preprocesses = {
        'CLIP' : preprocess_val,
        'CONCH' : conch_preprocess_val,
        'PLIP' : plip_preprocess_val,
        'MUSK': musk_preprocess_val
    }

    three_model_features = {
        'CLIP' : None,
        'CONCH' : None,
        'PLIP' : None,
        'MUSK': None
    }
    
    base_train_loader, base_val_loader, base_test_loader, dataset = get_all_dataloaders(cfg, preprocesses['CLIP'])
    features_list = []
    logits_list = []
    res = []
    models_name = []
    for k in cfg['selected_model'].split(','):
        temp_clip_model = models[k].cuda()
        temp_clip_model.eval()
        cfg['model'] = k

        if cfg['shots'] > 0:
            train_loader = build_data_loader(data_source=dataset.train_x, batch_size=64, is_train=False, tfm=preprocesses[k],
                                       shuffle=False)
        else:
            train_loader = None
        test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocesses[k],
                                       shuffle=False)
        val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocesses[k],
                                       shuffle=False)
        # shot_features, shot_labels, val_features, val_labels, test_features, test_labels, clip_prototypes
        three_model_features[k] = get_all_features(cfg, train_loader, val_loader, test_loader, dataset, temp_clip_model)
        test_labels = three_model_features[k][5]

        temp_clip_model = temp_clip_model.to('cpu')  # unload CLIP model from VRAM
        features_list.append(three_model_features[k][4].float())
        logits = prepare_objects(three_model_features[k][4], three_model_features[k][5], three_model_features[k][6])
        # logits_list.append(logits)
        acc = cls_acc(logits.cpu(), three_model_features[k][5])
        res.append(acc)
        models_name.append(k)
        # from TransCLIP_solver.TransCLIP import TransCLIP_solver
        # z, test_acc_at_best_val, acc_base_zs, acc = TransCLIP_solver(None, None, None, None, three_model_features[k][-3], three_model_features[k][-2], three_model_features[k][-1])
        # acc = cls_acc(z.cpu(), test_labels)
        # res.append(acc)
        
        logits_list.append(logits)
        
    z = OptimalTrans(features_list, test_labels, logits_list)
    acc = cls_acc(z.cpu(), test_labels)
    res.append(acc)
    print(cfg['dataset'])
    print(models_name)
    print(res)

    if args.method == 'TransCLIP':
        transclip_time = time.time()
        # from TransCLIP_solver_各自的y.TransCLIP import TransCLIP_solver
        # if cfg['shots'] == 0:
        #     TransCLIP_solver(cfg, three_model_features, args.dataset)
        # else:
        #     shot_features = shot_features.unsqueeze(0)
        #     TransCLIP_solver(shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
        #                      clip_prototypes)
        # print("feature time {}".format(str(time.time()-transclip_time)))

    elif args.method == 'On-top':

        from TransCLIP_solver.TransCLIP import TransCLIP_solver
        from TransCLIP_solver.bench_utils import prepare_for_bench

        test_features, test_labels, initial_prototypes, initial_logits, clip_prototypes = prepare_for_bench(args,
                                                                                                            test_features,
                                                                                                            test_labels,
                                                                                                            clip_prototypes)

        if cfg['shots'] == 0:
            TransCLIP_solver(shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
                             clip_prototypes, initial_prototypes, initial_logits)
        else:
            shot_features = shot_features.unsqueeze(0)
            TransCLIP_solver(shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
                             clip_prototypes, initial_prototypes, initial_logits)


    elif args.method == 'transductive_finetuning':
        device = clip_prototypes.device
        shot_features = shot_features.type(torch.FloatTensor).squeeze().to(device)
        val_features = val_features.type(torch.FloatTensor).squeeze().to(device)
        test_features = test_features.type(torch.FloatTensor).squeeze().to(device)
        clip_prototypes = clip_prototypes.type(torch.FloatTensor).squeeze().to(device)
        from baselines.runner import run_transductive_finetuning
        run_transductive_finetuning(cfg, shot_features, shot_labels, val_features, val_labels, test_features,
                                    test_labels,
                                    clip_prototypes)
    elif args.method == 'bdcspn':
        device = clip_prototypes.device
        shot_features = shot_features.type(torch.FloatTensor).squeeze().to(device)
        val_features = val_features.type(torch.FloatTensor).squeeze().to(device)
        test_features = test_features.type(torch.FloatTensor).squeeze().to(device)
        clip_prototypes = clip_prototypes.type(torch.FloatTensor).squeeze().to(device)
        from baselines.runner import run_bdcspn
        run_bdcspn(cfg, shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
                   clip_prototypes)
    elif args.method == 'laplacian_shot':
        device = clip_prototypes.device
        shot_features = shot_features.type(torch.FloatTensor).squeeze().to(device)
        val_features = val_features.type(torch.FloatTensor).squeeze().to(device)
        test_features = test_features.type(torch.FloatTensor).squeeze().to(device)
        clip_prototypes = clip_prototypes.type(torch.FloatTensor).squeeze().to(device)
        from baselines.runner import run_laplacian_shot
        run_laplacian_shot(cfg, shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
                           clip_prototypes)
    elif args.method == 'ptmap':
        device = clip_prototypes.device
        shot_features = shot_features.type(torch.FloatTensor).squeeze().to(device)
        val_features = val_features.type(torch.FloatTensor).squeeze().to(device)
        test_features = test_features.type(torch.FloatTensor).squeeze().to(device)
        clip_prototypes = clip_prototypes.type(torch.FloatTensor).squeeze().to(device)
        from baselines.runner import run_pt_map
        run_pt_map(cfg, shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
                   clip_prototypes)
    elif args.method == 'tim':
        device = clip_prototypes.device
        shot_features = shot_features.type(torch.FloatTensor).squeeze().to(device)
        val_features = val_features.type(torch.FloatTensor).squeeze().to(device)
        test_features = test_features.type(torch.FloatTensor).squeeze().to(device)
        clip_prototypes = clip_prototypes.type(torch.FloatTensor).squeeze().to(device)
        from baselines.runner import run_tim
        run_tim(cfg, shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
                clip_prototypes, version='adm')


if __name__ == '__main__':
    main()
