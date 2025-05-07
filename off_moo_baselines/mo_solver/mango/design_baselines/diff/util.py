"""Utilities used throughout the codebase."""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from wandb.sdk.wandb_run import Run
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
TASKNAME2TASK = {
    'dkitty': 'DKittyMorphology-Exact-v0',
    'ant': 'AntMorphology-Exact-v0',
    'tf-bind-8': 'TFBind8-Exact-v0',
    'tf-bind-10': 'TFBind10-Exact-v0',
    'superconductor': 'Superconductor-RandomForest-v0',
    # 'hopper': 'HopperController-Exact-v0',
    'nas': 'CIFARNAS-Exact-v0',
    'chembl': 'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0',
    # 'gfp': 'GFP-Transformer-v0',
}


def configure_gpu(use_gpu: bool, which_gpu: int) -> torch.device:
    """Set the GPU to be used for training."""
    if use_gpu:
        device = torch.device("cuda")
        # Only occupy one GPU, as in https://stackoverflow.com/questions/37893755/
        # tensorflow-set-cuda-visible-devices-within-jupyter
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu)
    else:
        device = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    return device


def set_seed(seed: Optional[int]) -> None:
    """Set the numpy, random, and torch random seeds."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)


def sorted_glob(*args, **kwargs) -> List[str]:
    """A sorted version of glob, to ensure determinism and prevent bugs."""
    return sorted(glob.glob(*args, **kwargs))


def parse_val_loss(filename: str) -> float:
    """Parse val_loss from the checkpoint filename."""
    start = filename.index("val_loss=") + len("val_loss=")
    try:
        end = filename.index("-v1.ckpt")
    except ValueError:
        end = filename.index(".ckpt")
    val_loss = float(filename[start:end])
    return val_loss


## REWEIGHTING
def adaptive_temp_v2(scores_np, q=None):
    """Calculate an adaptive temperature value based on the
    statistics of the scores array

    Args:

    scores_np: np.ndarray
        an array that represents the vectorized scores per data point

    Returns:

    temp: np.ndarray
        the scalar 90th percentile of scores in the dataset
    """

    inverse_arr = scores_np
    max_score = inverse_arr.max()
    scores_new = inverse_arr - max_score
    if q is None:
        quantile_ninety = np.quantile(scores_new, q=0.9)
    else:
        quantile_ninety = np.quantile(scores_new, q=q)
    return np.maximum(np.abs(quantile_ninety), 0.001)


def softmax(arr, temp=1.0):
    """Calculate the softmax using numpy by normalizing a vector
    to have entries that sum to one

    Args:

    arr: np.ndarray
        the array which will be normalized using a tempered softmax
    temp: float
        a temperature parameter for the softmax

    Returns:

    normalized: np.ndarray
        the normalized input array which sums to one
    """

    max_arr = arr.max()
    arr_new = arr - max_arr
    exp_arr = np.exp(arr_new / temp)
    return exp_arr / np.sum(exp_arr)


def get_weights(scores, base_temp=None, temp=None):
    """Calculate weights used for training a model inversion
    network with a per-sample reweighted objective

    Args:

    scores: np.ndarray
        scores which correspond to the value of data points in the dataset

    Returns:

    weights: np.ndarray
        an array with the same shape as scores that reweights samples
    """


    scores_np = scores#[:, 0]
    scores_np = (scores_np - np.min(scores_np)) / (np.max(scores_np) - np.min(scores_np))
    weights = 100*(1 - scores_np + 1e-3)

    # scores_np = scores#[:, 0]
    # hist, bin_edges = np.histogram(scores_np, bins=25)
    # hist = hist / np.sum(hist)

    # # if base_temp is None:
    # if temp == '90':
    #     base_temp = adaptive_temp_v2(scores_np, q=0.9)
    # elif temp == '75':
    #     base_temp = adaptive_temp_v2(scores_np, q=0.75)
    # elif temp == '50':
    #     base_temp = adaptive_temp_v2(scores_np, q=0.5)
    # else:
    #     raise RuntimeError("Invalid temperature")
    # softmin_prob = softmax(bin_edges[1:], temp=base_temp)
    # softmin_prob = 1 - softmin_prob #求最小值，越小越好

    # # provable_dist = softmin_prob * (hist / (hist + 1e-3))
    # # provable_dist = provable_dist / (np.sum(provable_dist) + 1e-7)


    # bin_indices = np.digitize(scores_np, bin_edges[1:])
    # hist_prob = hist[np.minimum(bin_indices, 24)]

    # # weights = provable_dist[np.minimum(bin_indices, 19)] / (hist_prob + 1e-7)
    # weights = softmin_prob[np.minimum(bin_indices, 24)] / (hist_prob + 1e-7)
    # weights = np.clip(weights, a_min=0.0, a_max=5.0)
    return weights.astype(np.float32)#[:, np.newaxis]

 

def get_weights_v1(Y):
    """
    对多目标数据集进行non-dominated sorting并分配权重
    
    参数:
    X: array-like, 输入变量
    Y: array-like, 多目标函数值
    
    返回:
    level_indices: list, 每个front对应的样本索引
    weights: array, 每个样本对应的权重
    """
    
    # 执行非支配排序
    nds = NonDominatedSorting()
    fronts = nds.do(Y)
    
    # 初始化
    n_samples = len(Y)
    weights = np.zeros(n_samples)
    num_fronts = len(fronts)
    
    # 为每个front分配权重：front等级越低（目标值越小（好）），权重越大
    for i, front_indices in enumerate(fronts):
        weight_value = (num_fronts - i) / num_fronts
        weights[front_indices] = weight_value
    
    return  weights

  
# # 获取front索引和权重
# front_indices, weights = get_level_weights(X, y)

# # 打印结果
# print("Number of fronts:", len(front_indices))
# for i, indices in enumerate(front_indices):
#     print(f"Front {i} has {len(indices)} samples with weight {weights[indices[0]]}")
# for i in range(len(front_indices)):
#     plt.scatter(y[front_indices[i]][:,0],y[front_indices[i]][:,1], alpha=0.1*weights[front_indices[i][0]])
 
 

def get_weights_v2(scores, base_temp=None, temp=None):
 
    scores_np = scores 
    bins_num = 10
    hist, bin_edges = np.histogram(scores, bins=bins_num)
    hist = hist / np.sum(hist)

    softmin_prob = softmax(bin_edges[1:], temp=0.9)
    # print(softmin_prob)
    softmin_prob = 1 - softmin_prob #求最小值，越小越好
    provable_dist = softmin_prob * (hist / (hist + 1e-3))
    provable_dist = provable_dist / (np.sum(provable_dist) + 1e-7)


    bin_indices = np.digitize(scores_np, bin_edges[1:])
    hist_prob = hist[np.minimum(bin_indices, bins_num-1)]

    weights = provable_dist[np.minimum(bin_indices, bins_num-1)] / (hist_prob + 1e-7)
    weights = np.clip(weights, a_min=0, a_max=5.0)
    return weights.astype(np.float32)[:, np.newaxis]


def get_weights_per_y(scores, base_temp=None, temp=None):
    """Calculate weights used for training a model inversion
    network with a per-sample reweighted objective

    Args:
    scores: np.ndarray
        scores which correspond to the value of data points in the dataset

    Returns:
    weights: np.ndarray
        an array with the same shape as scores that reweights samples
    """

    weights_list = []

    for i in range(scores.shape[1]):
        scores_np = scores[:, i]
        hist, bin_edges = np.histogram(scores_np, bins=20)
        hist = hist / np.sum(hist)

        if temp == '90':
            base_temp = adaptive_temp_v2(scores_np, q=0.9)
        elif temp == '75':
            base_temp = adaptive_temp_v2(scores_np, q=0.75)
        elif temp == '50':
            base_temp = adaptive_temp_v2(scores_np, q=0.5)
        else:
            raise RuntimeError("Invalid temperature")

        softmin_prob = softmax(bin_edges[1:], temp=base_temp)
        softmin_prob = 1 - softmin_prob #求最小值，越小越好


        provable_dist = softmin_prob * (hist / (hist + 1e-3))
        provable_dist = provable_dist / (np.sum(provable_dist) + 1e-7)
        # print('provable_dist', provable_dist)

        bin_indices = np.digitize(scores_np, bin_edges[1:])
        hist_prob = hist[np.minimum(bin_indices, 19)]

        weights = provable_dist[np.minimum(bin_indices, 19)] / (hist_prob + 1e-7)
        weights = np.clip(weights, a_min=0.0, a_max=5.0)
        weights_list.append(weights.astype(np.float32)[:, np.newaxis])

    return  np.hstack(weights_list)