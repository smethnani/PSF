import torch.nn as nn
import ot
class SlicedWassersteinDist(nn.Module):
    def __init__(self, batch_size, n_projections=100):
        super(SlicedWassersteinDist, self).__init__()
        self.bs = batch_size
        self.n_projections = n_projections
        
    def forward(self, P_batch, Q_batch):
      loss = 0
      for i in range(self.bs):
        loss += ot.sliced_wasserstein_distance(P_batch[i].view(-1, 1), Q_batch[i].view(-1, 1), n_projections=self.n_projections)
      return loss

class SphericalSlicedWassersteinDist(nn.Module):
    def __init__(self, batch_size):
        super(SphericalSlicedWassersteinDist, self).__init__()
        self.bs = batch_size
        
    def forward(self, P_batch, Q_batch):
      loss = 0
      for i in range(self.bs):
        loss += ot.sliced_wasserstein_sphere(P_batch[i].view(-1, 1), Q_batch[i].view(-1, 1))
      return loss