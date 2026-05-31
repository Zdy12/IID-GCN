import os
import numpy as np
import pandas as pd
import torch

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

def merge_bitext(folder):
    bitext_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            bitext_files.append(os.path.join(root, file))
    f_result = []
    for bitext_file in bitext_files:
        result = []
        with open(bitext_file, 'r') as file:
            for line in file:
                line = line.strip()
                row = line.split('\t')
                row_float = [float(element) for element in row]
                result.append(row_float)
        f_result.append(result)
    return f_result, bitext_files


def process_data(data, fixed_length):
    processed_data = []
    for patient_data in data:
        patient_data = torch.tensor(patient_data, dtype=torch.float32)
        if patient_data.ndimension() != 2:
            raise ValueError(
                f"Expected 2D tensor, got {patient_data.ndimension()}D tensor with shape {patient_data.shape}")
        num_rows, num_cols = patient_data.shape
        if num_rows < fixed_length:
            padding = fixed_length - num_rows
            patient_data = torch.cat([patient_data, torch.zeros(padding, num_cols)], dim=0)
        elif num_rows > fixed_length:
            patient_data = patient_data[:fixed_length]
        processed_data.append(patient_data)
    processed_data = [p_data.t() for p_data in processed_data]
    return torch.stack(processed_data)

desktop_path1 = os.path.expanduser('~/Desktop/TimeHOIAD/AD')
desktop_path2 = os.path.expanduser('~/Desktop/TimeHOIAD/CN')
desktop_path3 = os.path.expanduser('~/Desktop/TimeHOIAD/MCI')
# 数据加载
f_result1_AD, _ = merge_bitext(desktop_path1)
f_result1_CN, _ = merge_bitext(desktop_path2)
f_result1_MCI, _ = merge_bitext(desktop_path3)
# 数据预处理
T = 100
processed_data_AD = process_data(f_result1_AD, T)
processed_data_CN = process_data(f_result1_CN, T)
processed_data_MCI = process_data(f_result1_MCI, T)
print("processed_data_AD shape:", processed_data_AD.shape)
print("processed_data_CN shape:", processed_data_CN.shape)
print("processed_data_MCI shape:", processed_data_MCI.shape)

from itertools import combinations
from sxpid import SxPID
import numpy as np
import os
import torch


# ============================================================
# 1. 连续 fMRI 时间序列二值化
# ============================================================

def binarize_subject_ts(subject_ts, method="median", eps=1e-8):
    """
    subject_ts: shape = [n_roi, n_time]
        单个被试的 fMRI 时间序列。

    返回:
    X_bin: shape = [n_roi, n_time]
        二值化后的 0/1 时间序列。
    """

    X = np.asarray(subject_ts, dtype=np.float64)

    # 每个脑区沿时间维度做 z-score
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    Z = (X - mean) / (std + eps)

    if method == "median":
        # 每个脑区按自身中位数二值化
        thr = np.median(Z, axis=1, keepdims=True)
        X_bin = (Z > thr).astype(np.int8)

    elif method == "zero":
        # z-score 后按 0 二值化
        X_bin = (Z > 0).astype(np.int8)

    else:
        raise ValueError("method must be 'median' or 'zero'.")

    return X_bin


# ============================================================
# 2. 构造二元 SxPID 所需的联合分布
# ============================================================

def make_bivariate_dist(xi, xj):
    """
    xi, xj: shape = [n_time]
        两个脑区的二值化时间序列。

    返回:
    dist: SxPID.pid 所需的二元联合分布。
    """

    xi = np.asarray(xi, dtype=np.int8)
    xj = np.asarray(xj, dtype=np.int8)

    T = xi.shape[0]

    # 编码方式：
    # 00 -> 0
    # 01 -> 1
    # 10 -> 2
    # 11 -> 3
    joint = 2 * xi + xj

    counts = np.bincount(joint, minlength=4).astype(np.float64)
    probs = counts / T

    dist = {
        (0, 0, 0): probs[0],
        (0, 1, 1): probs[1],
        (1, 0, 2): probs[2],
        (1, 1, 3): probs[3],
    }

    return dist


def get_atom(avg, key):
    """
    安全读取 SxPID 的某个信息原子。
    如果某个 key 不存在，则返回 0。
    """
    if key in avg:
        return float(avg[key][0])
    else:
        return 0.0


# ============================================================
# 3. 计算单个被试的二元 PED
# ============================================================

def compute_one_subject_bivariate_ped(subject_ts, bin_method="median"):
    """
    subject_ts: shape = [90, 100]
        单个被试的 fMRI 时间序列。

    返回:
    result: shape = [90, 90, 3]

    result[:, :, 0] = redundancy matrix
    result[:, :, 1] = unique_total matrix
    result[:, :, 2] = synergy matrix

    其中 unique_total = unique_i + unique_j。
    """

    X_bin = binarize_subject_ts(subject_ts, method=bin_method)

    n_roi, n_time = X_bin.shape

    red_mat = np.zeros((n_roi, n_roi), dtype=np.float32)
    unique_mat = np.zeros((n_roi, n_roi), dtype=np.float32)
    syn_mat = np.zeros((n_roi, n_roi), dtype=np.float32)

    # 如果你后面想分析方向性唯一信息，可以保留这个矩阵：
    # unique_dir[i, j] 表示在边 (i, j) 中，节点 i 的唯一信息
    unique_dir = np.zeros((n_roi, n_roi), dtype=np.float32)

    for i, j in combinations(range(n_roi), 2):

        dist = make_bivariate_dist(X_bin[i], X_bin[j])

        _, avg = SxPID.pid(dist, verbose=0)

        # 冗余信息
        redundancy = get_atom(avg, ((1,), (2,)))

        # 两个脑区各自的唯一信息
        unique_i = get_atom(avg, ((1,),))
        unique_j = get_atom(avg, ((2,),))

        # 无向边上使用唯一信息总量
        unique_total = unique_i + unique_j

        # 协同信息
        synergy = get_atom(avg, ((1, 2),))

        # 对称赋值，因为现在二元脑网络先按无向图处理
        red_mat[i, j] = red_mat[j, i] = redundancy
        unique_mat[i, j] = unique_mat[j, i] = unique_total
        syn_mat[i, j] = syn_mat[j, i] = synergy

        # 保留方向性唯一信息
        unique_dir[i, j] = unique_i
        unique_dir[j, i] = unique_j

    result = np.stack(
        [red_mat, unique_mat, syn_mat],
        axis=-1
    )

    return result, unique_dir


# ============================================================
# 4. 计算一个类别中所有被试的二元 PED
# ============================================================

def compute_group_bivariate_ped(processed_data, group_name, out_dir, bin_method="median"):
    """
    processed_data: torch.Tensor, shape = [n_subject, 90, 100]

    返回:
    group_result: shape = [n_subject, 90, 90, 3]
    group_unique_dir: shape = [n_subject, 90, 90]
    """

    os.makedirs(out_dir, exist_ok=True)

    if isinstance(processed_data, torch.Tensor):
        data_np = processed_data.detach().cpu().numpy()
    else:
        data_np = np.asarray(processed_data)

    n_subject, n_roi, n_time = data_np.shape

    group_result = np.zeros((n_subject, n_roi, n_roi, 3), dtype=np.float32)
    group_unique_dir = np.zeros((n_subject, n_roi, n_roi), dtype=np.float32)

    print(f"\nStart computing {group_name}: {data_np.shape}")

    for s in range(n_subject):
        print(f"{group_name} subject {s + 1}/{n_subject}")

        subject_result, subject_unique_dir = compute_one_subject_bivariate_ped(
            data_np[s],
            bin_method=bin_method
        )

        group_result[s] = subject_result
        group_unique_dir[s] = subject_unique_dir

    save_path = os.path.join(out_dir, f"{group_name}_bivariate_PED.npz")

    np.savez_compressed(
        save_path,
        ped=group_result,
        unique_dir=group_unique_dir
    )

    print(f"{group_name} saved to: {save_path}")
    print(f"{group_name} PED shape: {group_result.shape}")

    return group_result, group_unique_dir


# ============================================================
# 5. 开始计算 AD / CN / MCI
# ============================================================

out_dir = os.path.expanduser("~/Desktop/TimeHOIAD_bivariate_PED")

ped_AD, unique_dir_AD = compute_group_bivariate_ped(
    processed_data_AD,
    group_name="AD",
    out_dir=out_dir,
    bin_method="median"
)

ped_CN, unique_dir_CN = compute_group_bivariate_ped(
    processed_data_CN,
    group_name="CN",
    out_dir=out_dir,
    bin_method="median"
)

ped_MCI, unique_dir_MCI = compute_group_bivariate_ped(
    processed_data_MCI,
    group_name="MCI",
    out_dir=out_dir,
    bin_method="median"
)

print("ped_AD shape:", ped_AD.shape)
print("ped_CN shape:", ped_CN.shape)
print("ped_MCI shape:", ped_MCI.shape)
