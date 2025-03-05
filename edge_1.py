import numpy as np
from torch import Tensor, LongTensor
from models.knn_points import knn_points
from models.knn import knn
import torch

def compute_edge_vectors(src_points, tgt_points, src_nodes,node_indices, filtered_node_num_nodes, s_perm_mat):
    from models.knn_points import knn_points

    batch_size, _, _ = src_points.size()
    num_nodes = filtered_node_num_nodes
    if node_indices.numel() != batch_size * num_nodes:
        raise ValueError("The number of node indices does not match the expected value.")
    k_nearest_neighbors = 3

    tgt_node_indices = []
    for b in range(batch_size):
        batch_tgt_indices = []
        current_batch_node_indices = node_indices[b]
        for row_idx in current_batch_node_indices:
            max_col_idx = torch.argmax(s_perm_mat[b, row_idx, :])
            batch_tgt_indices.append(max_col_idx.item())
        tgt_node_indices.append(batch_tgt_indices)
    tgt_node_indices = torch.tensor(tgt_node_indices)


    all_knn_indices = []

    for b in range(batch_size):
        batch_knn_indices = knn_points(src_nodes[b, :num_nodes], src_points[b], k=k_nearest_neighbors)
        all_knn_indices.append(batch_knn_indices)

    src_node_knn_indices = torch.stack(all_knn_indices, dim=0)  # 形状为 (batch_size, num_nodes, k_nearest_neighbors)

    neighbors_src = src_points[torch.arange(batch_size).view(-1, 1, 1), src_node_knn_indices]

    src_nodes_expanded = src_nodes.unsqueeze(2).repeat(1, 1, k_nearest_neighbors, 1)
    src_edges_tensor_true = neighbors_src - src_nodes_expanded.float()
    tgt_node_knn_indices = []
    for b in range(batch_size):
        # 提取当前批次的 s_perm_mat 和 src_node_knn_indices
        batch_s_perm_mat = s_perm_mat[b]
        batch_src_node_knn_indices = src_node_knn_indices[b]

        batch_tgt_indices = torch.empty((num_nodes, k_nearest_neighbors), dtype=torch.long, device=s_perm_mat.device)


        for m in range(num_nodes):
            node_knn_indices = batch_src_node_knn_indices[m]
            knn_rows = batch_s_perm_mat[node_knn_indices]
            node_tgt_indices = torch.argmax(knn_rows, dim=1)
            batch_tgt_indices[m] = node_tgt_indices
        tgt_node_knn_indices.append(batch_tgt_indices)

    tgt_node_knn_indices = torch.stack(tgt_node_knn_indices)
    node_tgt = tgt_points[torch.arange(batch_size).unsqueeze(1).repeat(1,num_nodes), tgt_node_indices]  # 形状为 (batch_size, num_nodes, 3)
    node_tgt = node_tgt.unsqueeze(2)
    neighbors_tgt = tgt_points[torch.arange(batch_size).unsqueeze(1).unsqueeze(2).repeat(1, num_nodes,
                                                                                         k_nearest_neighbors), tgt_node_knn_indices]  # 形状为 (batch_size, num_nodes, k_nearest_neighbors, 3)

    edges_tgt = neighbors_tgt - node_tgt

    src_knn = src_points[torch.arange(batch_size)[:, None, None].repeat(1, num_nodes,
                                                                        k_nearest_neighbors), src_node_knn_indices]  # Shape (batch_size, num_nodes, k_nearest_neighbors, 3)

    src_knn_diff = src_knn[:, :, :-1, :] - src_knn[:, :, 1:, :]  # Compute differences between consecutive k-NN points

    src_loop_edge = src_knn[:, :, 0, :] - src_knn[:, :, -1, :]  # Compute the loop edge

    src_loop_edge = src_loop_edge[:, :, None, :]  # Add an extra dimension for concatenation

    src_knn_edges = torch.cat((src_knn_diff, src_loop_edge), dim=2)  # Concatenate with the loop edge

    tgt_knn = tgt_points[torch.arange(batch_size)[:, None, None].repeat(1, num_nodes,
                                                                        k_nearest_neighbors), tgt_node_knn_indices]  # Shape (batch_size, num_nodes, k_nearest_neighbors, 3)

    tgt_knn_diff = tgt_knn[:, :, 1:, :]-tgt_knn[:, :, :-1, :]   # Compute differences between consecutive k-NN points

    tgt_loop_edge = tgt_knn[:, :, 0, :] - tgt_knn[:, :, -1, :]  # Compute the loop edge

    tgt_loop_edge = tgt_loop_edge[:, :, None, :]  # Add an extra dimension for concatenation

    tgt_knn_edges = torch.cat((tgt_knn_diff, tgt_loop_edge), dim=2)  # Concatenate with the loop edge
    # print(tgt_knn_edges.shape,tgt_knn_edges)



    return src_edges_tensor_true, edges_tgt, src_knn_edges, tgt_knn_edges


























































# gtfhnjmk,hmygtgregtfhgycjgytrfhtrefgfg


# # import numpy as np
# # from torch import Tensor, LongTensor
# # from models.knn_points import knn_points
# # from models.knn import knn
# # import torch
# # # import os
# # # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # does not affect results
# #
# #
# # def compute_edge_vectors(src_points, tgt_points, src_nodes,truncated_sampled_indices_tensor, filtered_node_num_nodes, s_perm_mat, k_nearest_neighbors=3):
# #
# #     # 确保输入参数有效
# #     batch_size, _, _ = src_points.size()  # 假设 src_points 的形状为 [batch_size1, 717num_points, 6]
# #     num_nodes = filtered_node_num_nodes
# #     k_nearest_neighbors = 3
# #     # ========================================================================================================================
# #
# # #     src_node_knn_indices = []
# # #     for b in range(batch_size):
# # #         for m in range(num_nodes):
# # #             # 获取当前节点的坐标
# # #             current_node = src_nodes[b, m].unsqueeze(0)  # 形状为 (1, 3)
# # #             # print(current_node,"current_node")
# # #             # 在当前批次的所有点中查询邻居
# # #             node_knn_indices_batch = knn_points(current_node,  # 查询点的坐标，形状为 (1, 3)
# # # src_points[b],  # 当前批次的所有点，形状为 (num_points, 3)
# # #                 k=k_nearest_neighbors).squeeze(0)  # 移除查询点的额外维度，结果为 (k,)
# # #             # print(node_knn_indices_batch,node_knn_indices_batch.shape,"node_knn_indices_batch")
# # #             src_node_knn_indices.append(node_knn_indices_batch)
# # #             # knn_coords = index_points( src_points[b], node_knn_indices_batch)
# # #
# # #             # 将索引列表堆叠成一个张量,jiashebiciB==1
# # #     src_node_knn_indices = torch.stack(src_node_knn_indices)
# # #
# # #     src_node_knn_indices=src_node_knn_indices.unsqueeze(0)
# #     # 存储每个批次中每个节点的邻居索引
# #     src_node_knn_indices_list = []  # 创建一个空列表来存储所有批次的邻居索引
# #
# #     for b in range(batch_size):
# #         batch_node_knn_indices = []  # 创建一个空列表来存储当前批次的邻居索引
# #         for m in range(num_nodes):
# #             # 获取当前节点的坐标，并添加一个维度以便进行 KNN 查询
# #             current_node = src_nodes[b, m].unsqueeze(0)  # 形状为 (1, 3)
# #             # 在当前批次的所有点中查询邻居（注意：这里可能应该使用整个点集而不是当前批次的点集）
# #             node_knn_indices_batch = knn_points(current_node, src_points[b], k=k_nearest_neighbors).squeeze(0)  # 结果为 (k,)
# #             batch_node_knn_indices.append(node_knn_indices_batch)  # 将当前节点的邻居索引添加到当前批次的列表中
# #         # 将当前批次的邻居索引堆叠成一个张量，并添加到总列表中
# #         batch_node_knn_indices = torch.stack(batch_node_knn_indices)  # 形状为 [num_nodes, k_nearest_neighbors]
# #         src_node_knn_indices_list.append(batch_node_knn_indices)  # 将当前批次的张量添加到总列表中
# #
# #     # 将所有批次的邻居索引堆叠成一个最终的张量
# #     src_node_knn_indices = torch.stack(src_node_knn_indices_list)  # 形状为 [batch_size, num_nodes, k_nearest_neighbors]
# #
# #     #print(src_node_knn_indices.shape)  # 输出形状以验证结果是否正确
# #     # print(src_node_knn_indices.shape,src_node_knn_indices)
# #     # print(src_points.shape,src_points)
# #
# #
# #     src_edges_list = []
# #     batch_size = src_nodes.size(0)
# #     num_nodes = src_nodes.size(1)
# #     for b in range(batch_size):
# #         for m in range(num_nodes):
# #             # node_src = src_points[b, node_indices[b, m]]  # 获取目标点云中对应的节点
# #             node_src=src_nodes[b, m]
# #             neighbors_src = src_points[b,src_node_knn_indices[b, m]]  # 获取目标点云中对应的邻居，NO排除节点本身
# #             edges_src = neighbors_src - node_src.unsqueeze(0)  # 计算边向量 (k, 3) - (1, 3) = (k, 3)
# #             # print(edges_src.shape,edges_src,"neighbors_src")
# #             src_edges_list.append(edges_src)  # 将边向量添加到列表中
# #     # print(src_edges_list,src_edges_list,"src_edges_list")
# #     # 将列表转换为张量并调整形状
# #     src_edges_tensor_true = torch.stack(src_edges_list).view(batch_size, num_nodes, k_nearest_neighbors, 3)  # (B, M', k, 3)
# #
# #     # print(src_edges_tensor_true.shape)  # 输出形状以验证结果是否正确
# #     #============================================
# #     # 使用 argmax 在最后一维上找到最大值的索引，结果形状为 (B, M)
# #     tgt_node_indices = torch.argmax(s_perm_mat, dim=2)
# #
# #     # 如果你确实需要结果形状为 (B, M, 1)，则可以使用 unsqueeze
# #     # tgt_node_indices = tgt_node_indices.unsqueeze(2)
# #
# #     # 现在 tgt_node_indices 的形状应该是 (B, M, 1)
# #
# #
# #     # ================================================
# #     tgt_node_knn_indices = []
# #     for b in range(batch_size):
# #         for m in range(num_nodes):
# #             node_knn_indices = src_node_knn_indices[b, m]  # Get the k-NN indices for the current source node
# #             node_tgt_indices = s_perm_mat[b, node_knn_indices].argmax(dim=1)  # Find the corresponding indices in the target point cloud
# #             tgt_node_knn_indices.append(node_tgt_indices)
# #
# #     tgt_node_knn_indices = torch.stack(tgt_node_knn_indices).view(batch_size, num_nodes, -1)  # (B, M', k)
# #     # print("tgt_points",tgt_points.device,tgt_node_indices.device)
# #
# #     #  Now compute the edge vectors in the target point cloud
# #     device = tgt_node_indices.device  # 获取tgt_node_indices所在的设备
# #     tgt_node_knn_indices = tgt_node_knn_indices.to(device)  # tgt_node_knn_indices转移到相同的设备上
# #     # print(tgt_node_knn_indices.shape)  # 输出形状以验证结果是否正确
# #
# #
# #     # 初始化一个空列表来收集所有的边向量
# #     # tgt_edges_list = []
# #     #
# #     # for b in range(batch_size):
# #     #     for m in range(num_nodes):
# #     #         node_tgt = tgt_points[b, tgt_node_indices[b, m]]  # 获取目标点云中对应的节点
# #     #         # neighbors_tgt = tgt_points[b, tgt_node_knn_indices[b, m, :-1]]  # 获取目标点云中对应的邻居，排除节点本身k-1
# #     #         neighbors_tgt = tgt_points[b, tgt_node_knn_indices[b, m]]  # 获取目标点云中对应的邻居，NO排除节点本身
# #     #         edges_tgt = neighbors_tgt - node_tgt.unsqueeze(0)  # 计算边向量 (k, 3) - (1, 3) = (k, 3)
# #     #         tgt_edges_list.append(edges_tgt)  # 将边向量添加到列表中
# #     #
# #     # # 将列表转换为张量并调整形状
# #     # tgt_edges_tensor = torch.stack(tgt_edges_list).view(batch_size, num_nodes, k_nearest_neighbors, 3)  # (B, M', k, 3)
# #
# #
# #     tgt_edges_list = [[] for _ in range(batch_size)]  # 创建一个空列表来存储所有批次的边向量
# #
# #     for b in range(batch_size):
# #         for m in range(num_nodes):
# #
# #             node_index = tgt_node_indices[b, m]  # 获取当前批次和节点的索引
# #             node_tgt = tgt_points[b, node_index].unsqueeze(0)  # 获取目标点云中对应的节点，并增加维度 (1, 3)
# #
# #             neighbors_indices = tgt_node_knn_indices[b, m]  # 获取当前节点的邻居索引
# #             print(neighbors_indices.shape, tgt_node_indices.shape, "534534")
# #
# #             neighbors_tgt = tgt_points[b, neighbors_indices]  # 获取目标点云中对应的邻居 (K, 3)
# #             edges_tgt = neighbors_tgt - node_tgt  # 计算边向量 (K, 3) - (1, 3) = (K, 3)，利用广播机制
# #
# #             tgt_edges_list[b].append(edges_tgt)  # 将边向量添加到当前批次的列表中
# #
# #     tgt_edges_tensor = torch.stack([torch.stack(batch_edges, dim=0) for batch_edges in tgt_edges_list])  # (B, M, K, 3)
# #
# #     # 可选：调整张量形状以匹配期望的输出 (B, M, K, 3)
# #
# #     # 如果已经正确构建了张量，则不需要这一步
# #
# #     # tgt_edges_tensor = tgt_edges_tensor.view(batch_size, num_nodes, k_nearest_neighbors, 3)
# #
# #
# #
# #
# #
# #
# #
# #
# #     src_knn_edges = []
# #     for b in range(batch_size):
# #         for m in range(num_nodes):
# #             knn = src_points[b, src_node_knn_indices[b, m, :]]  # 获取k近邻点（不包括节点本身）
# #             if knn.shape[0] > 1:  # 确保有足够的邻居来计算边
# #                 knn_edges = knn[1:] - knn[:-1]  # 计算连续k-NN点之间的边向量
# #
# #                 # 添加闭环边向量（从最后一个k-NN点到第一个k-NN点）
# #                 loop_edge = knn[0].unsqueeze(0) - knn[-1].unsqueeze(0)
# #                 knn_edges = torch.cat((knn_edges, loop_edge), dim=0)
# #                 src_knn_edges.append(knn_edges)
# #
# #     # Convert the list of edge tensors to a single tensor
# #     src_knn_edges_tensor = torch.nn.utils.rnn.pad_sequence([e.unsqueeze(0) for e in src_knn_edges], batch_first=True)
# #     src_knn_edges_tensor = src_knn_edges_tensor.view(batch_size, num_nodes, -1, src_points.size(2))  # Adjust the view according to your data dimensions
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #     # print(tgt_points,src_points,tgt_node_knn_indices,src_node_knn_indices)
# #
# #     # Do the same for the target point cloud
# #     tgt_knn_edges = []
# #     for b in range(batch_size):
# #         for m in range(num_nodes):
# #             knn_tgt = tgt_points[b, tgt_node_knn_indices[b, m, :]]  # Exclude the node itself
# #             # print(knn_tgt.shape,knn_tgt,"knn_tgt")
# #             if knn_tgt.shape[0] > 1:  # Check if there are enough neighbors to compute edges
# #                 knn_edges_tgt = knn_tgt[1:] - knn_tgt[:-1]  # Compute edge vectors between consecutive k-NN points
# #                 # knn_edges = np.vstack(knn_edges,knn[0]-knn[-1])
# #                 # 添加闭环边向量（从最后一个k-NN点到第一个k-NN点）
# #                 # print( knn_edges_tgt, "knn_edges_tgt")
# #                 loop_edge1 = knn_tgt[0].unsqueeze(0) - knn_tgt[-1].unsqueeze(0)
# #                 knn_edges_tgt = torch.cat((knn_edges_tgt, loop_edge1), dim=0)
# #                 tgt_knn_edges.append(knn_edges_tgt)
# #
# #     # Convert the list of edge tensors to a single tensor
# #     # tgt_knn_edges_tensor = torch.nn.utils.rnn.pad_sequence([e.unsqueeze(0) for e in tgt_knn_edges], batch_first=True).view(
# #     #     batch_size, num_nodes, -1, 3)  # (B, M', variable_length, 3)
# #     tgt_knn_edges_tensor = torch.nn.utils.rnn.pad_sequence([e.unsqueeze(0) for e in tgt_knn_edges], batch_first=True)
# #     tgt_knn_edges_tensor = tgt_knn_edges_tensor.view(batch_size, num_nodes, -1, tgt_points.size(2))  # Adjust the view according to your data dimensions
# #
# #     # print("tgt_edges_tensor", tgt_edges_tensor)
# #     #
# #     # print("src_knn_edges_tensor", src_knn_edges_tensor)
# #     # print("tgt_knn_edges_tensor", tgt_knn_edges_tensor)
# #     # print(src_edges_tensor_true,tgt_edges_tensor, src_knn_edges_tensor, tgt_knn_edges_tensor)
# #     return src_edges_tensor_true, tgt_edges_tensor, src_knn_edges_tensor, tgt_knn_edges_tensor
#
#
# #
# #
# #import numpy as np
# from torch import Tensor, LongTensor
# from models.knn_points import knn_points
# from models.knn import knn
# import torch
# # import os
# # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # does not affect results
#
#
# def compute_edge_vectors(src_points, tgt_points, src_nodes,truncated_sampled_indices_tensor, filtered_node_num_nodes, s_perm_mat, k_nearest_neighbors=3):
#     # print(src_points.shape,src_points)  torch.Size([1, 717, 3])
#     # print(tgt_points.shape,tgt_points) torch.Size([1, 717, 3])
#     # print(node_indices.shape,node_indices) torch.Size([1, 458]) 458
#     # print(filtered_node_num_nodes)
#     # print(s_perm_mat.shape,s_perm_mat)torch.Size([1, 717, 717])
#
#     # 确保输入参数有效
#     batch_size, _, _ = src_points.size()  # 假设 src_points 的形状为 [batch_size1, 717num_points, 6]
#     num_nodes = filtered_node_num_nodes
#     # 检查 node_indices 是否有效
#     if truncated_sampled_indices_tensor.numel() != batch_size * num_nodes:
#         raise ValueError("The number of node indices does not match the expected value.")
#
#     # 将 node_indices 平坦化以便索引
#     node_indices = truncated_sampled_indices_tensor.view(-1)
#
#     # 扩展 node_indices 以包括点云中的每个点的索引
#     expanded_indices = node_indices.unsqueeze(-1).repeat(1, 1, 3)  # 形状为 (batch_size, num_nodes, 3)
#
#     expanded_indices = expanded_indices.long()  # 确保索引是整数类型（如果它们不是的话）
#     # print(expanded_indices)
#     # 使用扩展的索引从 src_points 中选择节点
#     # 不重塑 src_points，直接使用高级索引
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     src_points = src_points.to(device)
#     expanded_indices = expanded_indices.to(device)
#     # ========================================================================================================================
#     # batch_size, num_nodes, _ = src_nodes.shape
#     k_nearest_neighbors = 3
#     # ========================================================================================================================
#
#     src_node_knn_indices_list = []  # 创建一个空列表来存储所有批次的邻居索引
#
#     for b in range(batch_size):
#         batch_node_knn_indices = []  # 创建一个空列表来存储当前批次的邻居索引
#         for m in range(num_nodes):
#             # 获取当前节点的坐标，并添加一个维度以便进行 KNN 查询
#             current_node = src_nodes[b, m].unsqueeze(0)  # 形状为 (1, 3)
#             # 在当前批次的所有点中查询邻居（注意：这里可能应该使用整个点集而不是当前批次的点集）
#             node_knn_indices_batch = knn_points(current_node, src_points[b], k=k_nearest_neighbors).squeeze(0)  # 结果为 (k,)
#             batch_node_knn_indices.append(node_knn_indices_batch)  # 将当前节点的邻居索引添加到当前批次的列表中
#         # 将当前批次的邻居索引堆叠成一个张量，并添加到总列表中
#         batch_node_knn_indices = torch.stack(batch_node_knn_indices)  # 形状为 [num_nodes, k_nearest_neighbors]
#         src_node_knn_indices_list.append(batch_node_knn_indices)  # 将当前批次的张量添加到总列表中
#
#     # 将所有批次的邻居索引堆叠成一个最终的张量
#     src_node_knn_indices = torch.stack(src_node_knn_indices_list)  # 形状为 [batch_size, num_nodes, k_nearest_neighbors]
#
#     src_edges_list = []
#     batch_size = src_nodes.size(0)
#     num_nodes = src_nodes.size(1)
#     for b in range(batch_size):
#         for m in range(num_nodes):
#             # node_src = src_points[b, node_indices[b, m]]  # 获取目标点云中对应的节点
#             node_src=src_nodes[b, m]
#             neighbors_src = src_points[b,src_node_knn_indices[b, m]]  # 获取目标点云中对应的邻居，NO排除节点本身
#             edges_src = neighbors_src - node_src.unsqueeze(0)  # 计算边向量 (k, 3) - (1, 3) = (k, 3)
#             # print(edges_src.shape,edges_src,"neighbors_src")
#             src_edges_list.append(edges_src)  # 将边向量添加到列表中
#     # print(src_edges_list,src_edges_list,"src_edges_list")
#     # 将列表转换为张量并调整形状
#     src_edges_tensor_true = torch.stack(src_edges_list).view(batch_size, num_nodes, k_nearest_neighbors, 3)  # (B, M', k, 3)
#     # =========================================
#
#     # 使用 argmax 在最后一维上找到最大值的索引，结果形状为 (B, M)
#     tgt_node_indices = torch.argmax(s_perm_mat, dim=2)
#     # print(tgt_node_indices.shape,tgt_node_indices)个广告广告广告广告广告广告广告广告广告广告广告
#
#
#     # ================================================
#     tgt_node_knn_indices = []
#     for b in range(batch_size):
#         for m in range(num_nodes):
#             node_knn_indices = src_node_knn_indices[b, m]  # Get the k-NN indices for the current source node
#             print(node_knn_indices,"tensor([162, 403, 218],")
#             node_tgt_indices = s_perm_mat[b, node_knn_indices].argmax(dim=1)  # Find the corresponding indices in the target point cloud
#             tgt_node_knn_indices.append(node_tgt_indices)
#
#     tgt_node_knn_indices = torch.stack(tgt_node_knn_indices).view(batch_size, num_nodes, -1)  # (B, M', k)
#     #============================
#     #  Now compute the edge vectors in the target point cloud
#     device = tgt_node_indices.device  # 获取tgt_node_indices所在的设备
#     tgt_node_knn_indices = tgt_node_knn_indices.to(device)  # tgt_node_knn_indices转移到相同的设备上
#
#     # 初始化一个空列表来收集所有的边向量
#     tgt_edges_list = []
#
#     for b in range(batch_size):
#         for m in range(num_nodes):
#             node_tgt = tgt_points[b, tgt_node_indices[b, m]]  # 获取目标点云中对应的节点
#             # neighbors_tgt = tgt_points[b, tgt_node_knn_indices[b, m, :-1]]  # 获取目标点云中对应的邻居，排除节点本身k-1
#             neighbors_tgt = tgt_points[b, tgt_node_knn_indices[b, m]]  # 获取目标点云中对应的邻居，NO排除节点本身
#             edges_tgt = neighbors_tgt - node_tgt.unsqueeze(0)  # 计算边向量 (k, 3) - (1, 3) = (k, 3)
#             tgt_edges_list.append(edges_tgt)  # 将边向量添加到列表中
#
#     # 将列表转换为张量并调整形状
#     tgt_edges_tensor = torch.stack(tgt_edges_list).view(batch_size, num_nodes, k_nearest_neighbors, 3)  # (B, M', k, 3)
#
#
#     src_knn_edges = []
#     for b in range(batch_size):
#         for m in range(num_nodes):
#             knn = src_points[b, src_node_knn_indices[b, m, :]]  # 获取k近邻点（不包括节点本身）
#             if knn.shape[0] > 1:  # 确保有足够的邻居来计算边
#                 knn_edges = knn[1:] - knn[:-1]  # 计算连续k-NN点之间的边向量
#
#                 # 添加闭环边向量（从最后一个k-NN点到第一个k-NN点）
#                 loop_edge = knn[0].unsqueeze(0) - knn[-1].unsqueeze(0)
#                 knn_edges = torch.cat((knn_edges, loop_edge), dim=0)
#                 src_knn_edges.append(knn_edges)
#
#     # Convert the list of edge tensors to a single tensor
#     src_knn_edges_tensor = torch.nn.utils.rnn.pad_sequence([e.unsqueeze(0) for e in src_knn_edges], batch_first=True)
#     src_knn_edges_tensor = src_knn_edges_tensor.view(batch_size, num_nodes, -1, src_points.size(2))  # Adjust the view according to your data dimensions
#
#     # print(tgt_points,src_points,tgt_node_knn_indices,src_node_knn_indices)
#
#     # Do the same for the target point cloud
#     tgt_knn_edges = []
#     for b in range(batch_size):
#         for m in range(num_nodes):
#             knn_tgt = tgt_points[b, tgt_node_knn_indices[b, m, :]]  # Exclude the node itself
#             # print(knn_tgt.shape,knn_tgt,"knn_tgt")
#             if knn_tgt.shape[0] > 1:  # Check if there are enough neighbors to compute edges
#                 knn_edges_tgt = knn_tgt[1:] - knn_tgt[:-1]  # Compute edge vectors between consecutive k-NN points
#                 # knn_edges = np.vstack(knn_edges,knn[0]-knn[-1])
#                 # 添加闭环边向量（从最后一个k-NN点到第一个k-NN点）
#                 # print( knn_edges_tgt, "knn_edges_tgt")
#                 loop_edge1 = knn_tgt[0].unsqueeze(0) - knn_tgt[-1].unsqueeze(0)
#                 knn_edges_tgt = torch.cat((knn_edges_tgt, loop_edge1), dim=0)
#                 tgt_knn_edges.append(knn_edges_tgt)
#
#     # Convert the list of edge tensors to a single tensor
#     # tgt_knn_edges_tensor = torch.nn.utils.rnn.pad_sequence([e.unsqueeze(0) for e in tgt_knn_edges], batch_first=True).view(
#     #     batch_size, num_nodes, -1, 3)  # (B, M', variable_length, 3)
#     tgt_knn_edges_tensor = torch.nn.utils.rnn.pad_sequence([e.unsqueeze(0) for e in tgt_knn_edges], batch_first=True)
#     tgt_knn_edges_tensor = tgt_knn_edges_tensor.view(batch_size, num_nodes, -1, tgt_points.size(2))  # Adjust the view according to your data dimensions
#
#
#
#     return src_edges_tensor_true, tgt_edges_tensor, src_knn_edges_tensor, tgt_knn_edges_tensor

