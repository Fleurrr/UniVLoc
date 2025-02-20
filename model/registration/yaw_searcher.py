import torch
import torch.nn as nn
import numpy as np
import pdb
import time

from model.registration.matching import pairwise_distance

class YawSearcher(nn.Module):
    def __init__(self, ):
        super(YawSearcher, self).__init__()
    
    @torch.no_grad()
    def calculate_distance(self, feature_q, feature_p):
        return torch.norm(feature_q - feature_p, dim=-1).mean(dim=-1)

    def shift_feature(self, D_q, D_p, step):
        N, D = D_q.size()
        if step >= 0:
            int_step = int(step)
            alpha = step - int_step
            shifted_D_q = (1 - alpha) * D_q.roll(int_step, 0) + alpha * D_q.roll(int_step + 1, 0)
        else:
            step = abs(step)
            int_step = int(step)
            alpha = step - int_step
            shifted_D_q = (1 - alpha) * D_q.roll(-int_step, 0) + alpha * D_q.roll(-int_step - 1, 0)
            
        return self.calculate_distance(D_p, shifted_D_q)

    def forward(self, D_q, D_p, max_step=60, step_size=0.5, prior_yaw=None, search_range=7):
        distances = []
        steps = []
        if prior_yaw is None:
            for step in np.arange(0, max_step, step_size):
                distances.append(self.shift_feature(D_q, D_p, step).item())
                steps.append(step)
            for step in np.arange(1, max_step + 1, step_size):
                distances.append(self.shift_feature(D_q, D_p, -step).item())
                steps.append(-step)
        else:
            for step in np.arange(0, search_range, step_size):
                distances.append(self.shift_feature(D_q, D_p, step + prior_yaw).item())
                steps.append(step + prior_yaw)
            for step in np.arange(1, search_range, step_size):
                distances.append(self.shift_feature(D_q, D_p, -step + prior_yaw).item())
                steps.append(-step + prior_yaw)           
        # [0, 1, 2, 3 ..., 59, -1, -2, ..., -60]
        min_distance = np.min(distances)
        optimal_step = np.argmin(distances)

        return steps[optimal_step], min_distance

    @torch.no_grad()
    def forward_roll(self, D_q, D_p, max_step=60, step_size=0.5, prior_yaw=None, search_range=7):
        feat_dists = torch.cdist(D_q, D_p, p=2).float()
        radius = feat_dists.shape[0]
        minimum_cost = 1.
        optimal_step = 1

        for i in range(1, radius // 2 + 1):
            m1 = feat_dists[i:, :radius - i ]
            m2 = feat_dists[radius - i:, :i]
            cost1, cost2 = torch.diag(m1).sum().item(), torch.diag(m2).sum().item()
            cost = (cost1 + cost2) / radius
            if cost < minimum_cost:
                minimum_cost = cost
                optimal_step = -i
        
        for i in range(radius // 2):
            m1 = feat_dists[:radius - i,  i:]
            m2 = feat_dists[:i, radius - i:]
            cost1, cost2 = torch.diag(m1).sum().item(), torch.diag(m2).sum().item()
            cost = (cost1 + cost2) / radius
            if cost < minimum_cost:
                minimum_cost = cost
                optimal_step = i
        return optimal_step, minimum_cost