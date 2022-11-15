
from torch import nn
import torch


class ALS(nn.Module):
    def __init__(self,n_users,n_items,dim=128):
        super(ALS, self).__init__()
        self.users = nn.Embedding(n_users, dim, max_norm=2)
        self.items = nn.Embedding(n_items, dim, max_norm=2)

    def forward(self, u,v):
        u = self.users(u)
        v = self.items(v)
        logit = torch.sum(u*v,axis=1)
        return logit