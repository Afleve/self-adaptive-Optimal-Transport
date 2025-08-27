from .OptimalTrans_utils import *
import time
import numpy as np


def OptimalTrans(vlm_features, vfm_features, true_labels=None, y_hats=None, cfg=None):
    
    if cfg is not None:
        import os
        os.makedirs(f'{cfg["root"]}', exist_ok=True)
    else:
        cfg = {
            'device' : 'cuda'
        }

    y_hats = [y_hat.to(cfg['device']) for y_hat in y_hats]
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.to(cfg['device'])
    K = y_hats[0].shape[1]
    num_samples =  y_hats[0].shape[0]
    features_list = [features.to(cfg['device']) for features in [*vlm_features, *vfm_features]] 
    # features_list = torch.cat(features_list, dim=-1) 
    # features_list = [torch.nn.functional.normalize(features_list, dim=-1)]

    iterations_to_try = list(range(0, 10))

    eps = 0.01
    S_iters = 10

    y_hats, z = init_z(y_hats, features_list, eps, S_iters, cfg, true_labels, initialize='avg')
    z_clone = z.clone()
    
    for _ in [0]:
        gmm_adapter_list = [None] * len(features_list)
        for i, features in enumerate(features_list):
            d = features.size(1)
            mu = init_mu(K, d, z_clone, features).to(cfg['device'])
            std_init = 1 / d
            std = init_sigma(d, std_init).to(cfg['device'])
            adapter = Gaussian(mu=mu, std=std).to(cfg['device'])
            gmm_adapter_list[i] = adapter
        gmm_likelihood_list = [None] * len(features_list)
        alpha = torch.ones(len(gmm_adapter_list)).to(cfg['device']) / (len(gmm_adapter_list))
        beta = torch.ones(len(y_hats)).to(cfg['device']) / (len(y_hats))
        
        for k in iterations_to_try:
            for i, adapter in enumerate(gmm_adapter_list):
                gmm_likelihood = adapter(features_list[i], no_exp=True)
                intermediate_temp = gmm_likelihood.clone()
                intermediate_temp -= torch.max(intermediate_temp, dim=1, keepdim=True)[0]
                intermediate_temp = torch.exp(1 / 10 * intermediate_temp)
                intermediate_temp = intermediate_temp / torch.sum(intermediate_temp, dim=1, keepdim=True)
                gmm_likelihood_list[i] = intermediate_temp


            fusion = torch.zeros(num_samples, K).to(cfg['device'])
            for i, P in enumerate(gmm_likelihood_list):
                fusion += P * alpha[i]
            for i, Y in enumerate(y_hats):
                fusion += Y * beta[i]

            z = sinkhorn(fusion, eps, S_iters)

            if len(gmm_adapter_list) > 1:
                alpha = update_alpha_per_source(z, gmm_likelihood_list)
            if len(y_hats) > 1:
                beta = update_alpha_per_source(z, y_hats)
                
            
            for i, adapter in enumerate(gmm_adapter_list):
                adapter = update_mu(adapter, features_list[i], z)
                adapter = update_sigma(adapter, features_list[i], z)
                gmm_adapter_list[i] = adapter

        if isinstance(true_labels, torch.Tensor):
            acc = cls_acc(z, true_labels)
        # print(alpha)
        # print(beta)
        
        # with open(f'{cfg["root"]}/{cfg["dataset"]}_log.txt', "a") as f: f.write(f"**** eps={eps:.4f}, it={S_iters:02d}, lambda={lambda_value:.4f}, k={k:02d} accuracy: {acc:.2f} ****\n")
        print(f"**** eps={eps:.4f}, it={S_iters:02d}, k={k:02d} accuracy: {acc:.2f} ****")

    return z
