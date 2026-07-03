import torch
import torch.nn.functional as F
import torch.nn as nn

@torch.no_grad()
def softmax_logits_chunk(logits_chunk, device):
    logits_chunk = logits_chunk.to(device=device, dtype=torch.float32, non_blocking=True)
    return torch.softmax(logits_chunk, dim=-1).cpu()

def empty_cuda_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()

@torch.no_grad()
def update_alpha_per_source(z, logits_list, device=None, chunk_size=5000):
    if len(logits_list) == 1:
        return torch.ones(1, dtype=torch.float32)
    z_prob = z.detach().cpu().float()
    scores = torch.zeros(len(logits_list), dtype=torch.float32)
    for i, logits in enumerate(logits_list):
        score = 0.0
        for start in range(0, z_prob.shape[0], int(chunk_size)):
            end = min(start + int(chunk_size), z_prob.shape[0])
            p = softmax_logits_chunk(logits[start:end], device or torch.device("cpu"))
            score += (z_prob[start:end] * p.float()).sum().item()
        scores[i] = score
        if device is not None:
            empty_cuda_cache(device)
    raw = 2.0 * torch.clamp(scores, min=1e-8)
    return raw / raw.sum().clamp_min(1e-12)


# def update_alpha_per_source(Q, P_list, eps=1e-8):
#     N = Q.shape[0]
#     K = Q.shape[1]
#     P_stack = torch.stack(P_list, dim=0)  # [M, N, K]
#     P_one_hot = F.one_hot(P_stack.argmax(dim=2), num_classes=K)
#     Q_one_hot = F.one_hot(Q.argmax(dim=1), num_classes=K)
    
#     H_batch_size, W_batch_size = 1000, 1000
#     z = torch.zeros(len(P_list), device=Q.device, dtype=Q.dtype) 
#     for m in range(len(P_list)):
#         for h in range(0, Q_one_hot.shape[0], H_batch_size):
#             for w in range(0, Q_one_hot.shape[1], W_batch_size):
#                 a = h + H_batch_size
#                 b = w + W_batch_size
#                 if a > Q_one_hot.shape[0]:
#                     a = Q_one_hot.shape[0]
#                 if b > Q_one_hot.shape[1]:
#                     b = Q_one_hot.shape[1]
#                 z[m] += torch.sum(Q_one_hot[h:a, w:b] * P_one_hot[m][h:a, w:b])
    
#     alpha_raw = (2.0 * torch.clamp(z, min=eps)) # [M]
#     alpha_raw = alpha_raw / alpha_raw.sum()
#     return alpha_raw  # shape: [M+1]


def calculate_loss(z, gmm_likelihood_list, y_hats, eps, alpha):
    obj_value = 0
    e = torch.eye(z.shape[0], device=z.device).bool()
    
    for i, likelihood in enumerate(gmm_likelihood_list):
        obj_value += (((z @ (1-likelihood).T)[e]) * alpha[i]).mean()

    for i, y_hat in enumerate(y_hats):
        obj_value += (((z @ (1-y_hat).T)[e]) * alpha[len(gmm_likelihood_list)+i]).mean()


    entropy = ((z@torch.log(z+1e-10).T)[e]).mean()
    obj_value += eps * entropy

    return obj_value.item()

@torch.no_grad()
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=10, tol=1e-5):
    out = out - out.max(dim=1, keepdim=True)[0]
    Q = torch.exp(out / epsilon).t()
    B = Q.shape[1]
    K = Q.shape[0]
    sum_Q = torch.sum(Q)
    Q /= sum_Q
    for _ in range(sinkhorn_iterations):
        Q_prev = Q.clone()
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
        err = torch.max(torch.abs(Q - Q_prev))
        if err < tol:
            break
    Q *= B
    return Q.t()


def update_z_wo_graph(gmm_likelihood, y_hat, z, lambda_value, labels=None, max_iter=5):
    few_shot = labels is not None
    if few_shot:
        shots_labels = F.one_hot(labels).float()
        z = torch.cat((z.clone(), shots_labels))

    num_samples = gmm_likelihood.size(0)
    for it in range(max_iter):
        intermediate = gmm_likelihood.clone()
        intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]
        intermediate = (y_hat ** lambda_value) * torch.exp(1 / 50 * intermediate)
        z[0:num_samples] = intermediate / torch.sum(intermediate, dim=1, keepdim=True)
    return z


def update_z(gmm_likelihood, y_hat, z, W, lambda_value, n_neighbors, labels=None, max_iter=5):
    few_shot = labels is not None
    if few_shot:
        shots_labels = F.one_hot(labels).float()
        z = torch.cat((z.clone(), shots_labels))

    num_samples = gmm_likelihood.size(0)
    for it in range(max_iter):
        intermediate = gmm_likelihood.clone()
        intermediate += (50 / (n_neighbors * 2)) * (W.T @ z + (W @ z[0:num_samples, :])[0:num_samples, :])
        # For numerical stability
        intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]
        pii = torch.exp(1 / 50 * intermediate)
        summ = torch.sum(pii, dim=1)

        intermediate = (y_hat ** lambda_value) * torch.exp(1 / 50 * intermediate)
        z[0:num_samples] = intermediate / torch.sum(intermediate, dim=1, keepdim=True)
    return z


def update_mu(adapter, query_features, z, support_features=None, labels=None, gamma_value=0):
    affinity_unlabeled = z
    n_query = affinity_unlabeled.size(0)
    few_shot = support_features is not None
    if few_shot:
        affinity_labeled = torch.nn.functional.one_hot(labels).float()
        n_support = affinity_labeled.size(0)

    weights = (1 / n_query) * affinity_unlabeled

    # Use einsum to compute the new_mu for each class in one pass
    new_mu = torch.einsum('ij,ik->jk', weights, query_features)

    if few_shot:
        weights = (gamma_value * 50 / n_support) * affinity_labeled
        new_mu += torch.einsum('ij,ik->jk', weights, support_features)

        new_mu /= (1 / n_query * torch.sum(
            affinity_unlabeled, dim=0).unsqueeze(
            -1) + gamma_value * 50 / n_support * torch.sum(
            affinity_labeled, dim=0).unsqueeze(-1))
    else:
        new_mu /= (1 / n_query * torch.sum(
            affinity_unlabeled, dim=0).unsqueeze(-1))
    new_mu = new_mu.unsqueeze(1)

    new_mu /= new_mu.norm(dim=-1, keepdim=True)

    adapter.mu = new_mu

    return adapter


def update_sigma(adapter, query_features, z, support_features=None, labels=None, gamma_value=0):
    affinity_unlabeled = z
    n_query = affinity_unlabeled.size(0)
    few_shot = support_features is not None
    if few_shot:
        affinity_labeled = torch.nn.functional.one_hot(labels).float()
        n_support = affinity_labeled.size(0)

    std = 0

    chunk_size = 1000  # Iterate over query_features in chunks to avoid large memory consumption

    for start_idx in range(0, n_query, chunk_size):
        end_idx = min(start_idx + chunk_size, n_query)
        query_features_chunk = query_features[start_idx:end_idx]

        # Compute the weighted sum of squared differences for the chunk
        chunk_result = (1 / n_query) * torch.einsum(
            'ij,ijk->k',
            affinity_unlabeled[start_idx:end_idx, :],
            # Use a chunk of affinity_unlabeled
            (query_features_chunk[:, None, :] - adapter.mu[None, :,
                                               0, :]) ** 2)

        # If this is the first chunk, initialize std; otherwise, accumulate
        if start_idx == 0:
            std = chunk_result
        else:
            std += chunk_result

    if few_shot and gamma_value > 0:
        # Iterate over query_features in chunks
        for start_idx in range(0, n_support, chunk_size):
            end_idx = min(start_idx + chunk_size, n_support)
            support_features_chunk = support_features[
                                     start_idx:end_idx]

            # Compute the weighted sum of squared differences for the chunk
            chunk_result = (gamma_value * 50 / n_support) * torch.einsum(
                'ij,ijk->k',
                affinity_labeled[start_idx:end_idx, :],
                # Use the relevant part of affinity_unlabeled
                (support_features_chunk[:, None, :] - adapter.mu[
                                                      None, :, 0,
                                                      :]) ** 2
            )

            std += chunk_result

        std /= (1 / n_query * torch.sum(
            affinity_unlabeled[:,
            :]) + gamma_value * 50 / n_support * torch.sum(
            affinity_labeled[:, :]))
    else:
        std /= (1 / n_query * torch.sum(
            affinity_unlabeled[:, :]))

    adapter.set_std(std)
    return adapter


def init_mu(K, d, z, query_features, support_features=None, support_labels=None):
    few_shot = support_features is not None
    if few_shot:
        support_labels_one_hot = F.one_hot(support_labels).float()
        num_shots = support_labels_one_hot.shape[0] // K
        t = support_features.cuda().squeeze().view(
                num_shots * K, d)
        mu = support_labels_one_hot.t() @ t
        mu = mu.unsqueeze(1)

    else:
        mu = torch.zeros(K, 1, d,
                         device=query_features.device)
        n_most_confident = 8
        topk_values, topk_indices = torch.topk(z, k=n_most_confident, dim=0)  # 8 pseudo-labels per class

        mask = torch.zeros_like(z).scatter_(0, topk_indices, 1)
        filtered_z = z * mask
        for c in range(K):
            class_indices = mask[:, c].nonzero().squeeze(1)
            class_features = query_features[class_indices]
            class_z = filtered_z[
                class_indices, c].unsqueeze(
                1)

            combined = class_features * class_z
            component_mean = combined[:n_most_confident].mean(dim=0)
            mu[c, 0, :] = component_mean
    mu /= mu.norm(dim=-1, keepdim=True)
    return mu


def init_sigma(d, std_init):
    std = (torch.eye(d).diag() * std_init)
    return std

def init_z(y_hats, features_list, eps, S_iters, cfg, true_labels, initialize='avg'):
    num_sources = len(y_hats)
    init_z = torch.zeros(y_hats[0].shape[0], y_hats[0].shape[1], device=y_hats[0].device)
    final_y_hats = []
    for i in range(num_sources):
        chunk_size = 1000
        # temp_z = torch.zeros_like(y_hats[i])
        # for start_idx in range(0, y_hats[i].shape[0], chunk_size):
        #     end_idx = min(start_idx + chunk_size, y_hats[i].shape[0])
        #     temp_z[start_idx:end_idx] = torch.nn.functional.softmax(y_hats[i][start_idx:end_idx], dim=-1)
        temp_z = torch.nn.functional.softmax(y_hats[i], dim=-1)
        init_z += temp_z
        final_y_hats.append(temp_z)
    z = init_z / num_sources
    if isinstance(true_labels, torch.Tensor):
        print("average init z: ", round(cls_acc(z, true_labels), 2))

    return final_y_hats, z

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

class Gaussian(nn.Module):
    def __init__(self, mu, std):
        super().__init__()
        self.mu = mu.clone()

        self.K, self.num_components, self.d = self.mu.shape
        self.std = std.clone()

        self.mixing = torch.ones(self.K, self.num_components, device=self.mu.device) / self.num_components

    def forward(self, x, get_components=False, no_exp=False):
        chunk_size = 1000
        N = x.shape[0]
        M, D = self.mu.shape[0], self.std.shape[0]

        intermediate = torch.empty((N, M), dtype=x.dtype, device=x.device)

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)

            intermediate[start_idx:end_idx] = -0.5 * torch.einsum('ijk,ijk->ij',
                                                                  (x[start_idx:end_idx][:, None, :] - self.mu[None, :,
                                                                                                      0, :]) ** 2,
                                                                  1 / self.std[None, None, :])

        if not no_exp:
            intermediate = torch.exp(intermediate)

        if get_components:
            return torch.ones_like(intermediate.unsqueeze(1))

        return intermediate

    def set_std(self, std):
        self.std = std



def prepare_objects(query_features, query_labels, clip_prototypes):

    query_features = query_features.cuda().float()
    query_labels = query_labels.cuda()
    clip_prototypes = clip_prototypes.cuda().float()

    if len(clip_prototypes.shape) == 3:  # use more than 1 template
        clip_prototypes = clip_prototypes[0]  # use only the first one

    clip_logits = 100 * query_features.cpu() @ clip_prototypes.cpu()

    acc = cls_acc(clip_logits, query_labels.cpu())
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))


    return clip_logits.cuda()
