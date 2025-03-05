#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math

import numpy as np
import torch
import torch.nn as nn
from utils.dcputil import quat2mat
import open3d
from models.edge_weight import compute_edge_weights
from models.edge_1 import compute_edge_vectors

class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dims
        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


def SVD(self, src, src_corr):
    # (batch, 3, n)
    batch_size = src.shape[0]
    src_centered = src - src.mean(dim=2, keepdim=True)
    src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

    H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous()).cpu()
    R = []
    for i in range(src.size(0)):
        u, s, v = torch.svd(H[i])
        r = torch.matmul(v, u.transpose(1, 0)).contiguous()
        r_det = torch.det(r).item()
        diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                          [0, 1.0, 0],
                                          [0, 0, r_det]]).astype('float32')).to(v.device)
        r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
        R.append(r)

    R = torch.stack(R, dim=0).cuda()
    t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
    if self.training:
        self.my_iter += 1
    return R, t.view(batch_size, 3)

def SVDslover1(src_o, tgt_o, s_perm_mat):
    """Compute rigid transforms between two point sets

    Args:
        src_o (torch.Tensor): (B, M, 3) points
        tgt_o (torch.Tensor): (B, N, 3) points
        s_perm_mat (torch.Tensor): (B, M, N)

    Returns:
        Transform R (B, 3, 3) t(B, 3) to get from src_o to tgt_o, i.e. T*src = tgt
    """
    weights1 = torch.sum(s_perm_mat, dim=2)
    weights_normalized1 = weights1[..., None] / (torch.sum(weights1[..., None], dim=1, keepdim=True) + 1e-5)
    weights2 = torch.sum(s_perm_mat, dim=1)  # 形状为 [B, N]
    weights_normalized2 = weights2[:, :, None] / torch.sum(weights2, dim=1, keepdim=True)[:, None, :]  # 形状为 [B, N, 1]
    centroid_src_o = torch.sum(src_o * weights_normalized1, dim=1)
    centroid_tgt_o = torch.sum(tgt_o * weights_normalized2, dim=1)
    src_o_centered = src_o - centroid_src_o[:, None, :]
    tgt_o_centered = tgt_o - centroid_tgt_o[:, None, :]
    cov = src_o_centered.transpose(-2, -1) @ (tgt_o_centered * weights_normalized2)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    R = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(R) > 0)

    # Compute translation (uncenter centroid)
    t = -R @ centroid_src_o[:, :, None] + centroid_tgt_o[:, :, None]

    return R, t.view(s_perm_mat.shape[0], 3)



def SVDslover(src_o, tgt_o, s_perm_mat, src1, src_corr1, src2, src_corr2):
    """Compute rigid transforms between two point sets

    Args:
        src_o (torch.Tensor): (B, M, 3) points
        tgt_o (torch.Tensor): (B, N, 3) points
        s_perm_mat (torch.Tensor): (B, M, N)
        src_edges: 源点云中节点与其k个最近邻居之间的边向量，形状为(B, M', k_nearest_neighbors, 3)。
        tgt_edges: 与源边向量对应的目标点云中的边向量，形状为(B, M', k_nearest_neighbors, 3)。
        src_knn_edges:
        tgt_knn_edges:
    Returns:
        Transform R (B, 3, 3) t(B, 3) to get from src_o to tgt_o, i.e. T*src = tgt
    """
    # print(src_edges)
    weights = torch.sum(s_perm_mat, dim=2)
    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + 1e-5)
    centroid_src_o = torch.sum(src_o * weights_normalized, dim=1)
    centroid_tgt_o = torch.sum(tgt_o * weights_normalized, dim=1)
    src_o_centered = src_o - centroid_src_o[:, None, :]
    tgt_o_centered = tgt_o - centroid_tgt_o[:, None, :]
    cov_point = src_o_centered.transpose(-2, -1) @ (tgt_o_centered * weights_normalized)

    src_centered1 = src1 - src1.mean(dim=2, keepdim=True)
    src_centered2 = src2 - src2.mean(dim=2, keepdim=True)

    src_corr_centered1 = src_corr1 - src_corr1.mean(dim=2, keepdim=True)
    src_corr_centered2 = src_corr2 - src_corr2.mean(dim=2, keepdim=True)

    H1 = torch.matmul(src_centered1, src_corr_centered1.transpose(2, 1).contiguous()).cpu()
    H2=  torch.matmul(src_centered2, src_corr_centered2.transpose(2, 1).contiguous()).cpu()
    cov_edge=H1+H2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make sure src_points and src_node_knn_indices are on the desired device

    cov_point = cov_point.to(device)

    cov_edge = cov_edge.to(device)
    # print(cov_point,"cov_point")
    # print(cov_edge,"cov_edge")
    cov= cov_point+cov_edge

    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    R = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(R) > 0)

    # Compute translation (uncenter centroid)
    centroid_src_o = centroid_src_o.to(device)
    centroid_tgt_o= centroid_tgt_o.to(device)
    t = -R @ centroid_src_o[:, :, None] + centroid_tgt_o[:, :, None]
    return R, t.view(s_perm_mat.shape[0], 3)










































#
# def SVDslover(src_o, tgt_o, s_perm_mat, src_edges, tgt_edges, src_knn_edges, tgt_knn_edges):
#     """Compute rigid transforms between two point sets
#
#     Args:
#         src_o (torch.Tensor): (B, M, 3) points
#         tgt_o (torch.Tensor): (B, N, 3) points
#         s_perm_mat (torch.Tensor): (B, M, N)
#         src_edges: 源点云中节点与其k个最近邻居之间的边向量，形状为(B, M', k_nearest_neighbors, 3)。
#         tgt_edges: 与源边向量对应的目标点云中的边向量，形状为(B, M', k_nearest_neighbors, 3)。
#         src_knn_edges:
#         tgt_knn_edges:
#     Returns:
#         Transform R (B, 3, 3) t(B, 3) to get from src_o to tgt_o, i.e. T*src = tgt
#     """
#     # print(src_edges)
#     weights = torch.sum(s_perm_mat, dim=2)
#     weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + 1e-5)
#     centroid_src_o = torch.sum(src_o * weights_normalized, dim=1)
#     centroid_tgt_o = torch.sum(tgt_o * weights_normalized, dim=1)
#     src_o_centered = src_o - centroid_src_o[:, None, :]
#     tgt_o_centered = tgt_o - centroid_tgt_o[:, None, :]
#     # print(src_edges)
#     # 计算加权平均的边向量
#     edge_weights = compute_edge_weights(src_edges, tgt_edges)
#     # print(edge_weights,"323")
#     # print(src_knn_edges,tgt_knn_edges)
#     edge_weights1 = compute_edge_weights(src_knn_edges, tgt_knn_edges)
#     # print(edge_weights,"3333")
#
#     weighted_src_edges1 = src_knn_edges * edge_weights1[..., None]  # (B, M', k_nearest_neighbors, 3)
#     weighted_src_edges = src_edges * edge_weights[..., None]  # (B, M', k_nearest_neighbors, 3)
#     weighted_src_edges =weighted_src_edges1+weighted_src_edges
#     weighted_tgt_edges = tgt_edges * edge_weights[..., None]  # (B, M', k_nearest_neighbors, 3)
#     weighted_tgt_edges1 = tgt_knn_edges * edge_weights1[..., None]  # (B, M', k_nearest_neighbors, 3)
#     weighted_tgt_edges =weighted_tgt_edges +weighted_tgt_edges1
#     # 对加权平均的边向量求和，得到每个节点的“平均边向量点”
#     avg_src_edge_points = torch.sum(weighted_src_edges, dim=2)  # (B, M', 3)
#     avg_tgt_edge_points = torch.sum(weighted_tgt_edges, dim=2)  # (B, M', 3)
#
#
#     # 现在可以计算质心和协方差
#     centroid_src_edge_points = torch.mean(avg_src_edge_points, dim=1)  # (B, 3)
#     centroid_tgt_edge_points = torch.mean(avg_tgt_edge_points, dim=1)  # (B, 3)
#     src_edge_points_centered = avg_src_edge_points - centroid_src_edge_points[:, None, :]
#     tgt_edge_points_centered = avg_tgt_edge_points - centroid_tgt_edge_points[:, None, :]
#
#     # 计算协方差
#     # cov_edge = src_edge_points_centered.transpose(-2, -1) @ tgt_edge_points_centered
#     # # 注意：这里的 cov_edge 可能不再是一个 3x3 矩阵，因为 M' 可能不等于 3
#     # # 可能需要进一步处理或解释这个协方差矩阵
#
#     cov_point = src_o_centered.transpose(-2, -1) @ (tgt_o_centered * weights_normalized)
#     # print(src_edge_points_centered,tgt_edge_points_centered,"ewfewhggggggggggggg")
#     cov_edge = src_edge_points_centered.transpose(-2, -1) @ tgt_edge_points_centered
#     # print(cov_edge)
#     # Determine the device you want to use. If CUDA is available, use it; otherwise, use CPU.
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     # Make sure src_points and src_node_knn_indices are on the desired device
#
#     cov_point = cov_point.to(device)
#
#     cov_edge = cov_edge.to(device)
#     # print(cov_point,"cov_point")
#     # print(cov_edge,"cov_edge")
#
#     cov= cov_point + 10e4*cov_edge
#
#     # print(cov_edge,"yyyyyyyyyyyyyy")
#     # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
#     # and choose based on determinant to avoid flips
#     u, s, v = torch.svd(cov, some=False, compute_uv=True)
#     rot_mat_pos = v @ u.transpose(-1, -2)
#     v_neg = v.clone()
#     v_neg[:, :, 2] *= -1
#     rot_mat_neg = v_neg @ u.transpose(-1, -2)
#     R = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
#     assert torch.all(torch.det(R) > 0)
#
#     # Compute translation (uncenter centroid)
#     centroid_src_o = centroid_src_o.to(device)
#     centroid_tgt_o= centroid_tgt_o.to(device)
#     t = -R @ centroid_src_o[:, :, None] + centroid_tgt_o[:, :, None]
#
#     return R, t.view(s_perm_mat.shape[0], 3)

#
# def RANSACSVDslover(src_o, tgt_o, s_perm_mat):
#     """Compute rigid transforms between two point sets with RANSAC
#
#     Args:
#         src_o (torch.Tensor): (B, M, 3) points
#         tgt_o (torch.Tensor): (B, N, 3) points
#         s_perm_mat (torch.Tensor): (B, M, N)
#
#     Returns:
#         Transform R (B, 3, 3) t(B, 3) to get from src_o to tgt_o, i.e. T*src = tgt
#     """
#     weights = torch.sum(s_perm_mat, dim=2)
#     weights_inlier = torch.where(weights==1)
#     import numpy as np
#
#     src_o0 = [src_o[i, list((weights_inlier[1][weights_inlier[0]==i]).cpu().numpy())].cpu().numpy()
#               for i in range(s_perm_mat.shape[0])]
#     tgt_o0 = [tgt_o[i, list((weights_inlier[1][weights_inlier[0]==i]).cpu().numpy())].cpu().numpy()
#               for i in range(s_perm_mat.shape[0])]
#     R = torch.zeros((s_perm_mat.shape[0], 3, 3)).to(s_perm_mat)
#     t = torch.zeros((s_perm_mat.shape[0], 3, 1)).to(s_perm_mat)
#     s_perm_mat_re = torch.zeros_like(s_perm_mat).to(s_perm_mat)
#     for i in range(len(src_o0)):
#         src_o0i = open3d.geometry.PointCloud()
#         tgt_o0i = open3d.geometry.PointCloud()
#         src_o0i.points = open3d.utility.Vector3dVector(src_o0[i])
#         tgt_o0i.points = open3d.utility.Vector3dVector(tgt_o0[i])
#         corr = open3d.utility.Vector2iVector(np.arange(src_o0[i].shape[0])[:,None].repeat(2, axis=1))
#         reg_result = open3d.registration.registration_ransac_based_on_correspondence(src_o0i, tgt_o0i, corr, 0.2)
#         R[i] = torch.from_numpy(reg_result.transformation[:3, :3]).to(s_perm_mat)
#         t[i] = torch.from_numpy(reg_result.transformation[:3, 3])[:,None].to(s_perm_mat)
#         corr_re = np.asarray(reg_result.correspondence_set)
#         s_perm_mat_re[i,corr_re[:,0]] = s_perm_mat[i,corr_re[:,0]]
#
#     return R, t.view(s_perm_mat.shape[0], 3), s_perm_mat_re
def RANSACSVDslover(src_o, tgt_o, s_perm_mat):
    """Compute rigid transforms between two point sets with RANSAC

    Args:
        src_o (torch.Tensor): (B, M, 3) points
        tgt_o (torch.Tensor): (B, N, 3) points
        s_perm_mat (torch.Tensor): (B, M, N)

    Returns:
        Transform R (B, 3, 3) t(B, 3) to get from src_o to tgt_o, i.e. T*src = tgt
    """
    weights = torch.sum(s_perm_mat, dim=2)
    weights_inlier = torch.where(weights==1)
    import numpy as np

    src_o0 = [src_o[i, list((weights_inlier[1][weights_inlier[0]==i]).cpu().numpy())].cpu().numpy()
              for i in range(s_perm_mat.shape[0])]
    tgt_o0 = [tgt_o[i, list((weights_inlier[1][weights_inlier[0]==i]).cpu().numpy())].cpu().numpy()
              for i in range(s_perm_mat.shape[0])]
    R = torch.zeros((s_perm_mat.shape[0], 3, 3)).to(s_perm_mat)
    t = torch.zeros((s_perm_mat.shape[0], 3, 1)).to(s_perm_mat)
    s_perm_mat_re = torch.zeros_like(s_perm_mat).to(s_perm_mat)
    for i in range(len(src_o0)):
        src_o0i = open3d.geometry.PointCloud()
        tgt_o0i = open3d.geometry.PointCloud()
        src_o0i.points = open3d.utility.Vector3dVector(src_o0[i])
        tgt_o0i.points = open3d.utility.Vector3dVector(tgt_o0[i])
        corr = open3d.utility.Vector2iVector(np.arange(src_o0[i].shape[0])[:,None].repeat(2, axis=1))
        reg_result = open3d.pipelines.registration.registration_ransac_based_on_correspondence(src_o0i, tgt_o0i, corr, 0.2)
        transformation_np = np.array(reg_result.transformation)
        R[i] = torch.from_numpy(transformation_np[:3, :3]).to(s_perm_mat)
        t[i] = torch.from_numpy(transformation_np[:3, 3])[:,None].to(s_perm_mat)
        corr_re = np.asarray(reg_result.correspondence_set)
        s_perm_mat_re[i,corr_re[:,0]] = s_perm_mat[i,corr_re[:,0]]


    return R, t.view(s_perm_mat.shape[0], 3), s_perm_mat_re
# def RANSACSVDslover(src_o, tgt_o, s_perm_mat):
#     """Compute rigid transforms between two point sets with RANSAC
#
#     Args:
#         src_o (torch.Tensor): (B, M, 3) points
#         tgt_o (torch.Tensor): (B, N, 3) points
#         s_perm_mat (torch.Tensor): (B, M, N)
#
#     Returns:
#         Transform R (B, 3, 3) t(B, 3) to get from src_o to tgt_o, i.e. T*src = tgt
#     """
#     weights = torch.sum(s_perm_mat, dim=2)
#     weights_inlier = torch.where(weights == 1)
#     import numpy as np
#
#     src_o0 = [src_o[i, list((weights_inlier[1][weights_inlier[0] == i]).cpu().numpy())].cpu().numpy()
#               for i in range(s_perm_mat.shape[0])]
#     tgt_o0 = [tgt_o[i, list((weights_inlier[1][weights_inlier[0] == i]).cpu().numpy())].cpu().numpy()
#               for i in range(s_perm_mat.shape[0])]
#     R = torch.zeros((s_perm_mat.shape[0], 3, 3)).to(s_perm_mat)
#     t = torch.zeros((s_perm_mat.shape[0], 3, 1)).to(s_perm_mat)
#     s_perm_mat_re = torch.zeros_like(s_perm_mat).to(s_perm_mat)
#     # Compute edge vectors between corresponding points
#     edge_vectors = []  # This will store edge vectors for each point pair
#     for i in range(len(src_o0)):
#         src_o0i = open3d.geometry.PointCloud()
#         tgt_o0i = open3d.geometry.PointCloud()
#         src_o0i.points = open3d.utility.Vector3dVector(src_o0[i])
#         tgt_o0i.points = open3d.utility.Vector3dVector(tgt_o0[i])
#         corr = open3d.utility.Vector2iVector(np.arange(src_o0[i].shape[0])[:, None].repeat(2, axis=1))
#         reg_result = open3d.registration.registration_ransac_based_on_correspondence(src_o0i, tgt_o0i, corr, 0.2)
#         R[i] = torch.from_numpy(reg_result.transformation[:3, :3]).to(s_perm_mat)
#         t[i] = torch.from_numpy(reg_result.transformation[:3, 3])[:, None].to(s_perm_mat)
#         corr_re = np.asarray(reg_result.correspondence_set)
#         s_perm_mat_re[i, corr_re[:, 0]] = s_perm_mat[i, corr_re[:, 0]]
#         # Compute edge vectors for each point pair and store them in edge_vectors list
#         edge_vectors.append([src_o0[i][j] - src_o0[i][k] for j, k in
#                              zip(corr[:-1], corr[1:])])  # Edge vectors from src to tgt for each point pair
#     # Use edge vectors to compute a rotation that aligns edge vectors of src and tgt point clouds
#     # ... Add code here to compute the rotation using edge vectors ...
#     return R, t.view(s_perm_mat.shape[0], 3), s_perm_mat_re






