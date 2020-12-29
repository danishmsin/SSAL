import torch 
import numpy as np
import os
import sys 
from utils.utils import *
from easydict import EasyDict as edict

def get_kNN(sim_distribution, feat_dict, k = 1):
    k_neighbors = torch.topk(torch.transpose(sim_distribution.cosines,0,1), k, dim = 1)
    idxs = k_neighbors[1]
    b_size = idxs.shape[0]
    k_neighbors = []
    labels_k_neighbors = []
    names_k_neighbors = []
    for i in range(b_size):
        l = idxs[i].cpu().data.numpy()
        k_neighbors.append(list(l))
        labels_k_neighbors.append(list(np.array(feat_dict.labels)[l]))
        names_k_neighbors.append([feat_dict.names[idx] for idx in list(l)])
    return k_neighbors, labels_k_neighbors, names_k_neighbors

def make_feat_dict_from_idx(feat_dict,idxs):
    feat_dict_idx = edict({})
    feat_dict_idx.feat_vec = feat_dict.feat_vec[idxs]
    feat_dict_idx.labels = [feat_dict.labels[idx] for idx in idxs]
    feat_dict_idx.names = [feat_dict.names[idx] for idx in idxs]
    feat_dict_idx.domain_identifier = [feat_dict.domain_identifier[idx] for idx in idxs]
    return feat_dict_idx

def get_k_farthest_neighbors(sim_distribution,feat_dict,K_farthest):
        sim_distribution.cosines = 1 - sim_distribution.cosines
        k_farthest, labels_k_farthest, names_k_farthest = get_kNN(sim_distribution, feat_dict, K_farthest)
        return k_farthest, labels_k_farthest, names_k_farthest

def do_source_weighting(loader, feat_dict,G,K_farthest,weight=0.8,aug = 2, only_for_poor = False, poor_class_list = None):
    farthest_classwise_examples_idx = []
    n_examples = len(feat_dict.domain_identifier)
    feat_dict.sample_weights = torch.tensor(np.ones(n_examples)).cuda()
    for idx, batch in enumerate(loader):
        img_vec = G(batch[0][aug])
        img_label = batch[1]
        idxs_label = [i for i, x in enumerate(feat_dict.labels) if x == img_label]
        feat_dict_label = make_feat_dict_from_idx(feat_dict,idxs_label)
        f_batch, sim_distribution = get_similarity_distribution(feat_dict_label,batch,G)
        k_farthest, labels_k_farthest, names_k_farthest = get_k_farthest_neighbors(sim_distribution,feat_dict_label,K_farthest)
        names_k_farthest = names_k_farthest[0] # 0 - since batch_size is 1 for assigning weights
        for name in names_k_farthest:
            idx_to_weigh = feat_dict.names.index(name)
            if img_label in poor_class_list:
                print(img_label.item())
                feat_dict.sample_weights[idx_to_weigh] = weight
            











