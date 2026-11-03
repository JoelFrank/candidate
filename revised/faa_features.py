"""FAA inductive bridge: ID vocabularies strictly fit on TRAIN; 0 is shared UNK feature,
but every unseen entity retains its own vertex index. No target report properties used.
"""
import torch
from torch import nn
class FAAFeatureEmbedder(nn.Module):
    def __init__(self, num_train_aircraft, num_train_parts, output_dim):
        super().__init__()
        self.u=nn.Embedding(num_train_aircraft+1,output_dim,padding_idx=0)
        self.v=nn.Embedding(num_train_parts+1,output_dim,padding_idx=0)
        self.pu=nn.Linear(output_dim,output_dim)
        self.pv=nn.Linear(output_dim,output_dim)
        nn.init.xavier_uniform_(self.u.weight);nn.init.xavier_uniform_(self.v.weight)
        with torch.no_grad():self.u.weight[0].zero_();self.v.weight[0].zero_()
    def forward(self,u_indices,v_indices):
        return self.pu(self.u(u_indices)),self.pv(self.v(v_indices))
