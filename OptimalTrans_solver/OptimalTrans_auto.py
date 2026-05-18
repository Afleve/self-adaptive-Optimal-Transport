from .OptimalTrans_utils import *
import time
import torch
import numpy as np

def evalute_test(cfg, test_features, test_labels, y_hats, gmm_adapter_list, alpha, beta):
    num_samples = test_features[0].shape[0]
    K = y_hats[0].shape[1]
    y_hats = [y_hat.to(cfg['device']) for y_hat in y_hats]
    test_features = [features.to(cfg['device']) for features in test_features] 
    test_labels = test_labels.to(cfg['device'])

    y_hats, z = init_z(y_hats, test_features, 0.01, 15, cfg, test_labels, initialize='avg')


    gmm_likelihood_list = []
    for i, adapter in enumerate(gmm_adapter_list):
        gmm_likelihood = adapter(test_features[i], no_exp=True)
        intermediate_temp = gmm_likelihood.clone()
        intermediate_temp -= torch.max(intermediate_temp, dim=1, keepdim=True)[0]
        intermediate_temp = torch.exp(1 / 50 * intermediate_temp)
        intermediate_temp = intermediate_temp / torch.sum(intermediate_temp, dim=1, keepdim=True)
        gmm_likelihood_list.append(intermediate_temp)
    fusion = torch.zeros(num_samples, K).to(cfg['device'])
    for i, P in enumerate(gmm_likelihood_list):
        fusion += P * alpha[i]
    for i, Y in enumerate(y_hats):
        fusion += Y * beta[i]
    acc = cls_acc(fusion, test_labels)
    print(f"**** Inductive Accuracy: {acc:.2f} ****")



def OptimalTrans(all_features, true_labels=None, y_hats=None, cfg=None):
    
    if cfg is not None:
        import os
        os.makedirs(f'{cfg["root"]}', exist_ok=True)
    else:
        cfg = {
            'device' : 'cuda',
            'setting' : 'Transductive'
        }

    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_reserved() / 1024**2
    start_time = time.time()

    y_hats = [y_hat.to(cfg['device']) for y_hat in y_hats]
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.to(cfg['device'])
    K = y_hats[0].shape[1]
    num_samples =  y_hats[0].shape[0]
    features_list = [features.to(cfg['device']).float() for features in all_features] 
    for features in features_list:
        print("features.shape = ", features.shape)

    print("K = ", K , "num_samples = ", num_samples)

    iterations_to_try = list(range(0, 10))

    eps = cfg.get('eps') if cfg.get('eps') else 0.01
    S_iters = 15

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
        alpha = torch.ones(len(gmm_adapter_list)).to(cfg['device']) / len(gmm_adapter_list)
        beta = torch.ones(len(y_hats)).to(cfg['device']) / len(y_hats)
        
        for k in iterations_to_try:
            for i, adapter in enumerate(gmm_adapter_list):
                gmm_likelihood = adapter(features_list[i], no_exp=True)
                intermediate_temp = gmm_likelihood.clone()
                intermediate_temp -= torch.max(intermediate_temp, dim=1, keepdim=True)[0]
                intermediate_temp = torch.exp(1 / 50 * intermediate_temp)
                intermediate_temp = intermediate_temp / torch.sum(intermediate_temp, dim=1, keepdim=True)
                gmm_likelihood_list[i] = intermediate_temp

            fusion = torch.zeros(num_samples, K).to(cfg['device'])
            for i, P in enumerate(gmm_likelihood_list):
                fusion += P * alpha[i]
            for i, Y in enumerate(y_hats):
                fusion += Y * beta[i]

            z = sinkhorn(fusion, eps, S_iters)
            
            if not len(gmm_likelihood_list) and len(y_hats)==1:
                break

            if len(gmm_adapter_list) > 1:
                alpha = update_alpha_per_source(z, gmm_likelihood_list)
            if len(y_hats) > 1:
                beta = update_alpha_per_source(z, y_hats)
            
            for i, adapter in enumerate(gmm_adapter_list):
                adapter = update_mu(adapter, features_list[i], z)
                adapter = update_sigma(adapter, features_list[i], z)
                gmm_adapter_list[i] = adapter
            acc = cls_acc(z, true_labels)

        if isinstance(true_labels, torch.Tensor):
            acc = cls_acc(z, true_labels)
        
        print(f"**** eps={eps:.4f}, it={S_iters:02d}, k={k:02d} accuracy: {acc:.2f} ****")
    
    print("T token time: ", time.time() - start_time)
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_reserved() / 1024**2
    print(f"GPU Memory: {mem_before:.2f} MB -> {mem_after:.2f} MB (delta: {mem_after - mem_before:.2f} MB)")
    
    if cfg['setting'] == 'Inductive':
        return gmm_adapter_list, alpha, beta

    return z, y_hats