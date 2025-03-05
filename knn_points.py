from typing import Tuple, Union
import torch
from torch import Tensor, LongTensor

from vision3d.utils.misc import load_ext
from models.knn import knn, keops_knn

ext_module = load_ext("vision3d.ext", ["knn_points"])


def knn_point1s(
    q_points: Tensor, s_points: Tensor, k: int, transposed: bool = False, return_distance: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Heap sort based kNN search for point cloud.

    Args:
        q_points (Tensor): The query points in shape of (B, N, 3) or (B, 3, N) if transposed.
        s_points (Tensor): The support points in shape of (B, M, 3) or (B, 3, M) if transposed.
        k (int): The number of neighbors.
        transposed (bool=False): If True, the points are in shape of (B, 3, N).
        return_distance (bool=False): If True, return the distances of the kNN.

    Returns:
        A Tensor of the distances of the kNN in shape of (B, N, k).
        A LongTensor of the indices of the kNN in shape of (B, N, k).
    """
    if transposed:
        q_points = q_points.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)
        s_points = s_points.transpose(1, 2)  # (B, M, 3) -> (B, 3, M)
    q_points = q_points.contiguous()
    s_points = s_points.contiguous()

    knn_distances = q_points.new_zeros(size=(q_points.shape[0], q_points.shape[1], k))  # (B, N, k)
    knn_indices = torch.zeros(size=(q_points.shape[0], q_points.shape[1], k), dtype=torch.long).cuda()  # (B, N, k)
    knn_indices= knn_indices.to("cpu")
    knn(q_points, s_points, knn_distances, knn_indices, k)

    if return_distance:
        return knn_distances, knn_indices

    return knn_indices


def knn_points(
    q_points: Tensor,
    s_points: Tensor,
    k: int,
    dilation: int = 1,
    distance_limit: float = None,
    return_distance: bool = None,
    remove_nearest: bool = True,
    transposed: bool = False,
    padding_mode: str = "nearest",
    padding_value: float = 1e10,
    squeeze: bool = False,
):
    """Compute the kNNs of the points in `q_points` from the points in `s_points`.

    Use KeOps to accelerate computation.

    Args:
        s_points (Tensor): coordinates of the support points, (*, C, N) or (*, N, C).
        q_points (Tensor): coordinates of the query points, (*, C, M) or (*, M, C).
        k (int): number of nearest neighbors to compute.
        dilation (int): dilation for dilated knn.
        distance_limit (float=None): if further than this radius, the neighbors are ignored according to `padding_mode`.
        return_distance (bool=False): whether return distances.
        remove_nearest (bool=True) whether remove the nearest neighbor (itself).
        transposed (bool=False): if True, the points shape is (*, C, N).
        padding_mode (str='nearest'): the padding mode for neighbors further than distance radius. ('nearest', 'empty').
        padding_value (float=1e10): the value for padding.
        squeeze (bool=False): if True, the distance and the indices are squeezed if k=1.

    Returns:
        knn_distances (Tensor): The distances of the kNNs, (*, M, k).
        knn_indices (LongTensor): The indices of the kNNs, (*, M, k).
    """
    if transposed:
        q_points = q_points.transpose(-1, -2)  # (*, C, N) -> (*, N, C)
        s_points = s_points.transpose(-1, -2)  # (*, C, M) -> (*, M, C)
    q_points = q_points.contiguous()
    s_points = s_points.contiguous()

    num_s_points = s_points.shape[-2]
    dilated_k = (k - 1)* dilation+ 1
    if remove_nearest:
        dilated_k += 1
    final_k = min(dilated_k, num_s_points)
    knn_distances, knn_indices = keops_knn(q_points, s_points, final_k)  # (*, N, k)
    if remove_nearest:
        knn_distances = knn_distances[..., 1:]
        knn_indices = knn_indices[..., 1:]
    if dilation > 1:
        knn_distances = knn_distances[..., ::dilation]
        knn_indices = knn_indices[..., ::dilation]
    knn_distances = knn_distances.contiguous()
    knn_indices = knn_indices.contiguous()
    if distance_limit is not None:
        assert padding_mode in ["nearest", "empty"]
        knn_masks = torch.ge(knn_distances, distance_limit)
        if padding_mode == "nearest":
            knn_distances = torch.where(knn_masks, knn_distances[..., :1], knn_distances)
            knn_indices = torch.where(knn_masks, knn_indices[..., :1], knn_indices)
        else:
            knn_distances[knn_masks] = padding_value
            knn_indices[knn_masks] = num_s_points
    if squeeze and k == 1:
        knn_distances = knn_distances.squeeze(-1)
        knn_indices = knn_indices.squeeze(-1)

    if return_distance:
        return knn_distances, knn_indices
    return knn_indices
