import numpy as np
import torch
import torch.nn as nn

from model.layers import build_dropout_layer
from model.registration.matching import pairwise_distance

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings

class RelativePositionEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k=3, reduction_a='max'):
        super(RelativePositionEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        embeddings = d_embeddings + a_embeddings

        return embeddings

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout=None):
        super(LearnablePositionalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)  # (L, D)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = build_dropout_layer(dropout)

    def forward(self, emb_indices):
        r"""Learnable Positional Embedding.

        `emb_indices` are truncated to fit the finite embedding space.

        Args:
            emb_indices: torch.LongTensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        emb_indices = emb_indices.view(-1)
        max_emd_indices = torch.full_like(emb_indices, self.num_embeddings - 1)
        emb_indices = torch.minimum(emb_indices, max_emd_indices)
        embeddings = self.embeddings(emb_indices)  # (*, D)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.view(*input_shape, self.embedding_dim)
        return embeddings

class RelativeAngleEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_a):
        super(RelativeAngleEmbedding, self).__init__()
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)
    
    @torch.no_grad()
    def get_embedding_indices(self, theta):
        row_indices= torch.arange(theta).unsqueeze(1)
        col_indices = torch.arange(theta).unsqueeze(0)
        angle_diff = torch.abs(row_indices - col_indices) * 360 /  theta
        a_indices = torch.where(angle_diff > 180, angle_diff - 360, angle_diff)
        a_indices = a_indices * self.factor_a
        return a_indices

    def forward(self, theta):
        a_indices = self.get_embedding_indices(theta).cuda()
        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        return a_embeddings

class RelativeDistanceEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, grid_size):
        super(RelativeDistanceEmbedding, self).__init__()
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.sigma_d = sigma_d
        self.grid_size = grid_size
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
    
    @torch.no_grad()
    def get_embedding_indices(self, N):
        row_indices= torch.arange(N).unsqueeze(1)
        col_indices = torch.arange(N).unsqueeze(0)
        dis_diff = torch.abs(row_indices - col_indices) * self.grid_size
        d_indices = dis_diff / self.sigma_d 
        return d_indices

    def forward(self, N):
        d_indices = self.get_embedding_indices(N).cuda()
        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)
        return d_embeddings