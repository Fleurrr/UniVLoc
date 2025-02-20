from typing import Optional
import torch
import numpy as np
import pdb
from model.registration.transformation import apply_transform

def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances

def index_select(data: torch.Tensor, index: torch.LongTensor, dim: int) -> torch.Tensor:
    r"""Advanced index select.

    Returns a tensor `output` which indexes the `data` tensor along dimension `dim`
    using the entries in `index` which is a `LongTensor`.

    Different from `torch.index_select`, `index` does not has to be 1-D. The `dim`-th
    dimension of `data` will be expanded to the number of dimensions in `index`.

    For example, suppose the shape `data` is $(a_0, a_1, ..., a_{n-1})$, the shape of `index` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.

    Args:
        data (Tensor): (a_0, a_1, ..., a_{n-1})
        index (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim: int

    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    output = data.index_select(dim, index.view(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)

    return output

# Extract correspondences
@torch.no_grad()
def extract_correspondences_from_scores(
    score_mat: torch.Tensor,
    mutual: bool = False,
    bilateral: bool = False,
    has_dustbin: bool = False,
    threshold: float = 0.0,
    return_score: bool = False,
):
    r"""Extract the indices of correspondences from matching scores matrix (max selection).

    Args:
        score_mat (Tensor): the logarithmic matching probabilities (N, M) or (N + 1, M + 1) according to `has_dustbin`
        mutual (bool = False): whether to get mutual correspondences.
        bilateral (bool = False), whether bilateral non-mutual matching, ignored if `mutual` is set.
        has_dustbin (bool = False): whether to use slack variables.
        threshold (float = 0): confidence threshold.
        return_score (bool = False): return correspondence scores.

    Returns:
        ref_corr_indices (LongTensor): (C,)
        src_corr_indices (LongTensor): (C,)
        corr_scores (Tensor): (C,)
    """
    score_mat = torch.exp(score_mat)
    ref_length, src_length = score_mat.shape

    ref_max_scores, ref_max_indices = torch.max(score_mat, dim=1)
    ref_indices = torch.arange(ref_length).cuda()
    ref_corr_scores_mat = torch.zeros_like(score_mat)
    ref_corr_scores_mat[ref_indices, ref_max_indices] = ref_max_scores
    ref_corr_masks_mat = torch.gt(ref_corr_scores_mat, threshold)

    if mutual or bilateral:
        src_max_scores, src_max_indices = torch.max(score_mat, dim=0)
        src_indices = torch.arange(src_length).cuda()
        src_corr_scores_mat = torch.zeros_like(score_mat)
        src_corr_scores_mat[src_max_indices, src_indices] = src_max_scores
        src_corr_masks_mat = torch.gt(src_corr_scores_mat, threshold)

        if mutual:
            corr_masks_mat = torch.logical_and(ref_corr_masks_mat, src_corr_masks_mat)
        else:
            corr_masks_mat = torch.logical_or(ref_corr_masks_mat, src_corr_masks_mat)
    else:
        corr_masks_mat = ref_corr_masks_mat

    if has_dustbin:
        corr_masks_mat = corr_masks_mat[:-1, :-1]

    ref_corr_indices, src_corr_indices = torch.nonzero(corr_masks_mat, as_tuple=True)

    if return_score:
        corr_scores = score_mat[ref_corr_indices, src_corr_indices]
        return ref_corr_indices, src_corr_indices, corr_scores
    else:
        return ref_corr_indices, src_corr_indices


@torch.no_grad()
def extract_correspondences_from_scores_threshold(
    scores_mat: torch.Tensor, threshold: float, has_dustbin: bool = False, return_score: bool = False
):
    r"""Extract the indices of correspondences from matching scores matrix (thresholding selection).

    Args:
        score_mat (Tensor): the logarithmic matching probabilities (N, M) or (N + 1, M + 1) according to `has_dustbin`
        threshold (float = 0): confidence threshold
        has_dustbin (bool = False): whether to use slack variables
        return_score (bool = False): return correspondence scores

    Returns:
        ref_corr_indices (LongTensor): (C,)
        src_corr_indices (LongTensor): (C,)
        corr_scores (Tensor): (C,)
    """
    scores_mat = torch.exp(scores_mat)
    if has_dustbin:
        scores_mat = scores_mat[:-1, :-1]
    masks = torch.gt(scores_mat, threshold)
    ref_corr_indices, src_corr_indices = torch.nonzero(masks, as_tuple=True)

    if return_score:
        corr_scores = scores_mat[ref_corr_indices, src_corr_indices]
        return ref_corr_indices, src_corr_indices, corr_scores
    else:
        return ref_corr_indices, src_corr_indices


@torch.no_grad()
def extract_correspondences_from_scores_topk(
    scores_mat: torch.Tensor, k: int, has_dustbin: bool = False, largest: bool = True, return_score: bool = False
):
    r"""Extract the indices of correspondences from matching scores matrix (global top-k selection).

    Args:
        score_mat (Tensor): the scores (N, M) or (N + 1, M + 1) according to `has_dustbin`.
        k (int): top-k.
        has_dustbin (bool = False): whether to use slack variables.
        largest (bool = True): whether to choose the largest ones.
        return_score (bool = False): return correspondence scores.

    Returns:
        ref_corr_indices (LongTensor): (C,)
        src_corr_indices (LongTensor): (C,)
        corr_scores (Tensor): (C,)
    """
    corr_indices = scores_mat.view(-1).topk(k=k, largest=largest)[1]
    ref_corr_indices = corr_indices // scores_mat.shape[1]
    src_corr_indices = corr_indices % scores_mat.shape[1]
    if has_dustbin:
        ref_masks = torch.ne(ref_corr_indices, scores_mat.shape[0] - 1)
        src_masks = torch.ne(src_corr_indices, scores_mat.shape[1] - 1)
        masks = torch.logical_and(ref_masks, src_masks)
        ref_corr_indices = ref_corr_indices[masks]
        src_corr_indices = src_corr_indices[masks]

    if return_score:
        corr_scores = scores_mat[ref_corr_indices, src_corr_indices]
        return ref_corr_indices, src_corr_indices, corr_scores
    else:
        return ref_corr_indices, src_corr_indices


@torch.no_grad()
def extract_correspondences_from_feats(
    ref_feats: torch.Tensor,
    src_feats: torch.Tensor,
    mutual: bool = False,
    bilateral: bool = False,
    return_feat_dist: bool = False,
):
    r"""Extract the indices of correspondences from feature distances (nn selection).

    Args:
        ref_feats (Tensor): features of reference point cloud (N, C).
        src_feats (Tensor): features of source point cloud (M, C).
        mutual (bool = False): whether to get mutual correspondences.
        bilateral (bool = False), whether bilateral non-mutual matching, ignored if `mutual` is set.
        return_feat_dist (bool = False): return feature distances.

    Returns:
        ref_corr_indices (LongTensor): (C,)
        src_corr_indices (LongTensor): (C,)
        corr_feat_dists (Tensor): (C,)
    """
    feat_dists_mat = pairwise_distance(ref_feats, src_feats)

    ref_corr_indices, src_corr_indices = extract_correspondences_from_scores(
        -feat_dists_mat,
        mutual=mutual,
        has_dustbin=False,
        bilateral=bilateral,
    )

    if return_feat_dist:
        corr_feat_dists = feat_dists_mat[ref_corr_indices, src_corr_indices]
        return ref_corr_indices, src_corr_indices, corr_feat_dists
    else:
        return ref_corr_indices, src_corr_indices

# Point correspondences
@torch.no_grad()
def get_point_correspondences(
    ref_nodes: torch.Tensor,
    src_nodes: torch.Tensor,
    transform: torch.Tensor,
    pos_radius: float,
):
    ref_nodes = apply_transform(ref_nodes, transform)
    distances = torch.sqrt(pairwise_distance(ref_nodes[...,:2], src_nodes[...,:2])) 
    mask = distances < pos_radius
    qry_indices, pos_indices = torch.nonzero(mask, as_tuple=True)
    gt_point_corr_indices = torch.stack((qry_indices, pos_indices), dim=-1)
    gt_point_corr_distances = distances[mask]
    # np.save('./ref_nodes.npy', ref_nodes.detach().cpu().numpy())
    # np.save('./src_nodes.npy', src_nodes.detach().cpu().numpy())
    # np.save('./point_corr_indices.npy', gt_point_corr_indices.detach().cpu().numpy())
    # pdb.set_trace()
    return gt_point_corr_indices, gt_point_corr_distances


# Patch correspondences
@torch.no_grad()
def get_node_correspondences(
    ref_nodes: torch.Tensor,
    src_nodes: torch.Tensor,
    ref_knn_points: torch.Tensor,
    src_knn_points: torch.Tensor,
    transform: torch.Tensor,
    pos_radius: float,
    ref_masks: Optional[torch.Tensor] = None,
    src_masks: Optional[torch.Tensor] = None,
    ref_knn_masks: Optional[torch.Tensor] = None,
    src_knn_masks: Optional[torch.Tensor] = None,
):
    r"""Generate ground-truth superpoint/patch correspondences.

    Each patch is composed of at most k nearest points of the corresponding superpoint.
    A pair of points match if the distance between them is smaller than `self.pos_radius`.

    Args:
        ref_nodes: torch.Tensor (M, 3)
        src_nodes: torch.Tensor (N, 3)
        ref_knn_points: torch.Tensor (M, K, 3)
        src_knn_points: torch.Tensor (N, K, 3)
        transform: torch.Tensor (4, 4)
        pos_radius: float
        ref_masks (optional): torch.BoolTensor (M,) (default: None)
        src_masks (optional): torch.BoolTensor (N,) (default: None)
        ref_knn_masks (optional): torch.BoolTensor (M, K) (default: None)
        src_knn_masks (optional): torch.BoolTensor (N, K) (default: None)

    Returns:
        corr_indices: torch.LongTensor (C, 2)
        corr_overlaps: torch.Tensor (C,)
    """
    src_nodes = apply_transform(src_nodes, transform)
    src_knn_points = apply_transform(src_knn_points, transform)

    # generate masks
    if ref_masks is None:
        ref_masks = torch.ones(size=(ref_nodes.shape[0],), dtype=torch.bool).cuda()
    if src_masks is None:
        src_masks = torch.ones(size=(src_nodes.shape[0],), dtype=torch.bool).cuda()
    if ref_knn_masks is None:
        ref_knn_masks = torch.ones(size=(ref_knn_points.shape[0], ref_knn_points.shape[1]), dtype=torch.bool).cuda()
    if src_knn_masks is None:
        src_knn_masks = torch.ones(size=(src_knn_points.shape[0], src_knn_points.shape[1]), dtype=torch.bool).cuda()

    node_mask_mat = torch.logical_and(ref_masks.unsqueeze(1), src_masks.unsqueeze(0))  # (M, N)

    # filter out non-overlapping patches using enclosing sphere
    ref_knn_dists = torch.linalg.norm(ref_knn_points - ref_nodes.unsqueeze(1), dim=-1)  # (M, K)
    ref_knn_dists.masked_fill_(~ref_knn_masks, 0.0)
    ref_max_dists = ref_knn_dists.max(1)[0]  # (M,)
    src_knn_dists = torch.linalg.norm(src_knn_points - src_nodes.unsqueeze(1), dim=-1)  # (N, K)
    src_knn_dists.masked_fill_(~src_knn_masks, 0.0)
    src_max_dists = src_knn_dists.max(1)[0]  # (N,)
    dist_mat = torch.sqrt(pairwise_distance(ref_nodes, src_nodes))  # (M, N)
    intersect_mat = torch.gt(ref_max_dists.unsqueeze(1) + src_max_dists.unsqueeze(0) + pos_radius - dist_mat, 0)
    intersect_mat = torch.logical_and(intersect_mat, node_mask_mat)
    sel_ref_indices, sel_src_indices = torch.nonzero(intersect_mat, as_tuple=True)

    # select potential patch pairs
    ref_knn_masks = ref_knn_masks[sel_ref_indices]  # (B, K)
    src_knn_masks = src_knn_masks[sel_src_indices]  # (B, K)
    ref_knn_points = ref_knn_points[sel_ref_indices]  # (B, K, 3)
    src_knn_points = src_knn_points[sel_src_indices]  # (B, K, 3)

    point_mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))  # (B, K, K)

    # compute overlaps
    dist_mat = pairwise_distance(ref_knn_points, src_knn_points)  # (B, K, K)
    dist_mat.masked_fill_(~point_mask_mat, 1e12)
    point_overlap_mat = torch.lt(dist_mat, pos_radius ** 2)  # (B, K, K)
    ref_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-1), dim=-1).float()  # (B,)
    src_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-2), dim=-1).float()  # (B,)
    ref_overlaps = ref_overlap_counts / ref_knn_masks.sum(-1).float()  # (B,)
    src_overlaps = src_overlap_counts / src_knn_masks.sum(-1).float()  # (B,)
    overlaps = (ref_overlaps + src_overlaps) / 2  # (B,)

    overlap_masks = torch.gt(overlaps, 0)
    ref_corr_indices = sel_ref_indices[overlap_masks]
    src_corr_indices = sel_src_indices[overlap_masks]
    corr_indices = torch.stack([ref_corr_indices, src_corr_indices], dim=1)
    corr_overlaps = overlaps[overlap_masks]

    return corr_indices, corr_overlaps


@torch.no_grad()
def node_correspondences_to_dense_correspondences(
    ref_knn_points,
    src_knn_points,
    ref_knn_indices,
    src_knn_indices,
    node_corr_indices,
    transform,
    matching_radius,
    ref_knn_masks=None,
    src_knn_masks=None,
    return_distance=False,
):
    if ref_knn_masks is None:
        ref_knn_masks = torch.ones_like(ref_knn_indices)
    if src_knn_masks is None:
        src_knn_masks = torch.ones_like(src_knn_indices)

    src_knn_points = apply_transform(src_knn_points, transform)
    ref_node_corr_indices = node_corr_indices[:, 0]  # (P,)
    src_node_corr_indices = node_corr_indices[:, 1]  # (P,)
    ref_node_corr_knn_indices = ref_knn_indices[ref_node_corr_indices]  # (P, K)
    src_node_corr_knn_indices = src_knn_indices[src_node_corr_indices]  # (P, K)
    ref_node_corr_knn_points = ref_knn_points[ref_node_corr_indices]  # (P, K, 3)
    src_node_corr_knn_points = src_knn_points[src_node_corr_indices]  # (P, K, 3)
    ref_node_corr_knn_masks = ref_knn_masks[ref_node_corr_indices]  # (P, K)
    src_node_corr_knn_masks = src_knn_masks[src_node_corr_indices]  # (P, K)
    dist_mat = torch.sqrt(pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points))  # (P, K, K)
    corr_mat = torch.lt(dist_mat, matching_radius)
    mask_mat = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
    corr_mat = torch.logical_and(corr_mat, mask_mat)  # (P, K, K)
    batch_indices, row_indices, col_indices = torch.nonzero(corr_mat, as_tuple=True)  # (C,) (C,) (C,)
    ref_corr_indices = ref_node_corr_knn_indices[batch_indices, row_indices]
    src_corr_indices = src_node_corr_knn_indices[batch_indices, col_indices]
    corr_indices = torch.stack([ref_corr_indices, src_corr_indices], dim=1)
    if return_distance:
        corr_distances = dist_mat[batch_indices, row_indices, col_indices]
        return corr_indices, corr_distances
    else:
        return corr_indices


@torch.no_grad()
def get_node_overlap_ratios(
    ref_points,
    src_points,
    ref_knn_points,
    src_knn_points,
    ref_knn_indices,
    src_knn_indices,
    node_corr_indices,
    transform,
    matching_radius,
    ref_knn_masks,
    src_knn_masks,
    eps=1e-5,
):
    corr_indices = node_correspondences_to_dense_correspondences(
        ref_knn_points,
        src_knn_points,
        ref_knn_indices,
        src_knn_indices,
        node_corr_indices,
        transform,
        matching_radius,
        ref_knn_masks=ref_knn_masks,
        src_knn_masks=ref_knn_masks,
    )
    unique_ref_corr_indices = torch.unique(corr_indices[:, 0])
    unique_src_corr_indices = torch.unique(corr_indices[:, 1])
    ref_overlap_masks = torch.zeros(ref_points.shape[0] + 1).cuda()  # pad for following indexing
    src_overlap_masks = torch.zeros(src_points.shape[0] + 1).cuda()  # pad for following indexing
    ref_overlap_masks.index_fill_(0, unique_ref_corr_indices, 1.0)
    src_overlap_masks.index_fill_(0, unique_src_corr_indices, 1.0)
    ref_knn_overlap_masks = index_select(ref_overlap_masks, ref_knn_indices, dim=0)  # (N', K)
    src_knn_overlap_masks = index_select(src_overlap_masks, src_knn_indices, dim=0)  # (M', K)
    ref_knn_overlap_ratios = (ref_knn_overlap_masks * ref_knn_masks).sum(1) / (ref_knn_masks.sum(1) + eps)
    src_knn_overlap_ratios = (src_knn_overlap_masks * src_knn_masks).sum(1) / (src_knn_masks.sum(1) + eps)
    return ref_knn_overlap_ratios, src_knn_overlap_ratios


@torch.no_grad()
def get_node_occlusion_ratios(
    ref_points,
    src_points,
    ref_knn_points,
    src_knn_points,
    ref_knn_indices,
    src_knn_indices,
    node_corr_indices,
    transform,
    matching_radius,
    ref_knn_masks,
    src_knn_masks,
    eps=1e-5,
):
    ref_knn_overlap_ratios, src_knn_overlap_ratios = get_node_overlap_ratios(
        ref_points,
        src_points,
        ref_knn_points,
        src_knn_points,
        ref_knn_indices,
        src_knn_indices,
        node_corr_indices,
        transform,
        matching_radius,
        ref_knn_masks,
        src_knn_masks,
        eps=eps,
    )
    ref_knn_occlusion_ratios = 1.0 - ref_knn_overlap_ratios
    src_knn_occlusion_ratios = 1.0 - src_knn_overlap_ratios
    return ref_knn_occlusion_ratios, src_knn_occlusion_ratios