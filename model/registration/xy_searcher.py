import torch
import torch.nn as nn
import numpy as np
import pdb

class XYSearcher(nn.Module):
    def __init__(self, ):
        super(XYSearcher, self).__init__()
    
    @torch.no_grad()
    def calculate_distance(self, feature_q, feature_p):
        return torch.norm(feature_q - feature_p, dim=-1).mean(dim=-1)

    def shift_feature(self, D_q, D_p, step):
        N, D = D_q.size()
        if step >= 0:
            int_step = int(step)
            alpha = step - int_step
            shifted_D_q = alpha * D_q.roll(-int_step, 0) + (1 - alpha) * D_q.roll(-int_step - 1, 0)
            return self.calculate_distance(shifted_D_q[:N - int_step, :], D_p[:N - int_step, :])
        else:
            step = abs(step)
            int_step = int(step)
            alpha = step - int_step
            shifted_D_p = alpha * D_p.roll(-int_step, 0) + (1 - alpha) * D_p.roll(-int_step - 1, 0)
            return self.calculate_distance(D_q[:N - int_step, :], shifted_D_p[: N - int_step, :])

    def forward(self, D_q, D_p, max_step=60, step_size=1):
        # distances = torch.zeros(int(max_step * 2 / step_size), dtype=torch.float32, device=D_q.device)
        distances = []
        steps = []
        for step in np.arange(0, max_step, step_size):
            distances.append(self.shift_feature(D_q, D_p, step).item())
            steps.append(step)
        for step in np.arange(1, max_step + 1, step_size):
            distances.append(self.shift_feature(D_q, D_p, -step).item())
            steps.append(-step)
        # [0, 1, 2, 3 ..., 59, -1, -2, ..., -60]
        min_distance = np.min(distances)
        optimal_step = np.argmin(distances)

        return steps[optimal_step], min_distance