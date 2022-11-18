import math

import torch
import torch.nn as nn
from geoopt import ManifoldParameter

from manifolds import Lorentz, PseudoHyperboloid
from utils.euclidean import givens_rotations, givens_reflection
import torch.nn.functional as F

pdist = nn.PairwiseDistance(p=2)
torch.cuda.set_device(2)

def lorentz_linear(x, weight, scale, bias=None):
    x = x @ weight.transpose(-2, -1)
    if bias is not None:
        x = x + bias
    time = x.narrow(-1, 0, 1).sigmoid() * scale + 1.1
    x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
    x_narrow = x_narrow / ((x_narrow * x_narrow).sum(dim=-1, keepdim=True) / (time * time - 1)).sqrt()
    x = torch.cat([time, x_narrow], dim=-1)
    return x

class LorentzE(torch.nn.Module):
    def __init__(self, d, dim, max_scale, max_norm, margin):
        super(LorentzE, self).__init__()
        self.manifold = Lorentz(max_norm=max_norm)
        self.emb_entity = ManifoldParameter(self.manifold.random_normal((len(d.entities), dim), std=1./math.sqrt(dim)), manifold=self.manifold)
        self.relation_bias = nn.Parameter(torch.zeros((len(d.relations), dim)))
        self.relation_transform = nn.Parameter(torch.empty(len(d.relations), dim, dim))
        nn.init.kaiming_uniform_(self.relation_transform)
        self.scale = nn.Parameter(torch.ones(()) * max_scale, requires_grad=False)
        self.margin = margin
        self.bias_head = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.bias_tail = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx,eval_mode=False):
        h = self.emb_entity[u_idx]
        t = self.emb_entity[v_idx]
        r_bias = self.relation_bias[r_idx]
        r_transform = self.relation_transform[r_idx]
        h = lorentz_linear(h.unsqueeze(1), r_transform, self.scale, r_bias.unsqueeze(1)).squeeze(1)
        neg_dist = (self.margin + 2 * self.manifold.cinner(h.unsqueeze(1), t).squeeze(1))
        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]

class UltraE(torch.nn.Module):
    def __init__(self, d, dim, max_scale, max_norm, margin, dist='e'):
        super(UltraE, self).__init__()
        self.time_dim = 2
        self.dim = dim
        self.beta = torch.tensor([-1.0]).cuda()
        self.manifold = PseudoHyperboloid(space_dim = dim - self.time_dim, time_dim=self.time_dim, beta=self.beta) 

        self.emb_entity = nn.Parameter(torch.randn((len(d.entities), dim)))
        self.relation_boost = nn.Parameter((1e-5)*F.softplus(torch.randn((len(d.relations), self.time_dim))))
        self.relation_rot_left = nn.Parameter(2*torch.rand((len(d.relations), dim))-1.0)
        self.relation_rot_right = nn.Parameter(2*torch.rand((len(d.relations), dim))-1.0)

        self.margin = margin
        self.bias_head = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.bias_tail = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.loss = torch.nn.BCEWithLogitsLoss()
        # fixed curvature
        # beta = -torch.ones((1)).float()
        # self.beta = nn.Parameter(beta, requires_grad=False)
        self.dist = dist

    def forward(self, u_idx, r_idx, v_idx,eval_mode=False):
        head = self.emb_entity[u_idx]
        tail = self.emb_entity[v_idx]

        rel_boost = F.softplus(self.relation_boost[r_idx])
        r_rot_right = self.relation_rot_right[r_idx]
        r_rot_left = self.relation_rot_left[r_idx]
        head = self.manifold.expmap0(head,self.beta)
        # assert self.manifold._check_point_on_manifold(head, self.beta)
        tail = self.manifold.expmap0(self.manifold.proj_tan0(tail.view(-1,tail.shape[-1]),self.beta),self.beta)
        # assert self.manifold._check_point_on_manifold(tail, self.beta)
        tail = tail.view(head.shape[0],-1,tail.shape[-1])
        
        head = givens_rotations(r_rot_right, head)
        assert self.manifold._check_point_on_manifold(head, self.beta)

        C = torch.diag_embed((1+rel_boost**2).sqrt())
        S = torch.diag_embed(rel_boost)
        I = torch.eye(self.dim - 2*self.time_dim)
        boost = torch.zeros((rel_boost.shape[0],self.dim ,self.dim)).cuda()
        boost[:,0:self.time_dim,0:self.time_dim] = C
        boost[:,self.dim-self.time_dim:,self.dim-self.time_dim:] = C
        boost[:,0:self.time_dim,self.dim-self.time_dim:] = -S
        boost[:,self.dim-self.time_dim:,0:self.time_dim] = -S
        boost[:,self.time_dim:self.dim-self.time_dim,self.time_dim:self.dim-self.time_dim] = I
        head = torch.matmul(boost, head.unsqueeze(-1)).squeeze(-1)
        assert self.manifold._check_point_on_manifold(head, self.beta)

        head = givens_reflection(r_rot_left, head)
        assert self.manifold._check_point_on_manifold(head, self.beta)
        # dist = self.sqdist(head,tail).view(head.shape[0],-1)
        dist = self.intrinsic_dist(head,tail)
        neg_dist = self.margin - dist
        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]

    def sqdist(self,h,t):
        l = h.repeat_interleave(t.shape[1],dim=0)
        r = t.view(-1,t.shape[-1])
        return self.manifold.sqdist(l,r,self.beta)

    def extrinsic_dist(self,h,t):
        l = h.repeat_interleave(t.shape[1],dim=0)
        r = t.view(-1,t.shape[-1])
        cdist = pdist(l, r).view(h.shape[0], t.shape[1]) 
        return cdist

    def intrinsic_dist(self, h, t):
        l = h.repeat_interleave(t.shape[1],dim=0)
        r = t.view(-1,t.shape[-1])
        cdist_1 = self.manifold.manhattan_sqdist(l, r, self.beta).view(h.shape[0], t.shape[1]) 
        cdist_2 = self.manifold.manhattan_sqdist(r, l, self.beta).view(h.shape[0], t.shape[1]) 
        return torch.min(cdist_1,cdist_2)

class RotE(torch.nn.Module):
    def __init__(self, d, dim, max_scale, max_norm, margin):
        super(RotE, self).__init__()
        self.time_dim = 4
        self.dim = dim
        self.beta = torch.tensor([-1.0]).cuda()
    
        self.emb_entity = nn.Parameter((1e-5)*torch.randn((len(d.entities), dim)))
        self.relation_rot= nn.Parameter(2*torch.rand((len(d.relations), dim))-1.0)
        self.relation_rot_center = nn.Parameter(torch.randn((len(d.relations), dim)))
        self.relation_trans = nn.Parameter(torch.randn((len(d.relations), dim)))

        self.margin = margin
        self.bias_head = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.bias_tail = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx,eval_mode=False):
        head = self.emb_entity[u_idx]
        tail = self.emb_entity[v_idx]
        r_rot = self.relation_rot[r_idx]
        r_trans = self.relation_trans[r_idx]
        r_center = self.relation_rot_center[r_idx]
        head = givens_rotations(r_rot, head+r_center)- r_center + r_trans
        dist = self.cdist(head,tail)
        neg_dist = self.margin - dist
        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]

    def cdist(self,h,t):
        l = h.repeat_interleave(t.shape[1],dim=0)
        r = t.view(-1,t.shape[-1])
        cdist = pdist(l, r).view(h.shape[0], t.shape[1]) 
        return cdist


class RotH(torch.nn.Module):
    def __init__(self, d, dim, max_scale, max_norm, margin):
        super(RotH, self).__init__()
        self.dim = dim
        self.beta = torch.tensor([-1.0]).cuda()
        # self.manifold = PseudoHyperboloid(space_dim = dim - self.time_dim, time_dim=self.time_dim, beta=self.beta) 

        self.emb_entity = nn.Parameter((1e-5)*torch.randn((len(d.entities), dim)))
        self.relation_rot= nn.Parameter(2*torch.rand((len(d.relations), dim))-1.0)
        self.relation_rot_center = nn.Parameter(torch.randn((len(d.relations), dim)))
        self.relation_trans = nn.Parameter(torch.randn((len(d.relations), dim)))

        self.margin = margin
        self.bias_head = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.bias_tail = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx,eval_mode=False):
        head = self.emb_entity[u_idx]
        tail = self.emb_entity[v_idx]
        r_rot = self.relation_rot[r_idx]
        r_trans = self.relation_trans[r_idx]
        r_center = self.relation_rot_center[r_idx]
        head = expmap0(head, self.beta.abs())
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(queries[:, 1]), lhs)

        res2 = mobius_add(res1, rel2, c)
        head = givens_rotations(r_rot, head+r_center)- r_center + r_trans
        dist = self.cdist(head,tail)
        neg_dist = self.margin - dist
        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]

    def cdist(self,h,t):
        l = h.repeat_interleave(t.shape[1],dim=0)
        r = t.view(-1,t.shape[-1])
        cdist = pdist(l, r).view(h.shape[0], t.shape[1]) 
        return cdist


# class RotH(torch.nn.Module):
#     def __init__(self, d, dim, max_scale, max_norm, margin):
#         super(RotH, self).__init__()
#         self.time_dim = 4
#         self.dim = dim
#         self.beta = torch.tensor([-1.0]).cuda()
#         self.manifold = PseudoHyperboloid(space_dim = dim - self.time_dim, time_dim=self.time_dim, beta=self.beta) 

#         self.emb_entity = nn.Parameter((1e-5)*torch.randn((len(d.entities), dim)))
#         self.relation_rot= nn.Parameter(2*torch.rand((len(d.relations), dim))-1.0)
#         self.relation_rot_center = nn.Parameter(torch.randn((len(d.relations), dim)))
#         self.relation_trans = nn.Parameter(torch.randn((len(d.relations), dim)))

#         self.margin = margin
#         self.bias_head = torch.nn.Parameter(torch.zeros(len(d.entities)))
#         self.bias_tail = torch.nn.Parameter(torch.zeros(len(d.entities)))
#         self.loss = torch.nn.BCEWithLogitsLoss()

#     def forward(self, u_idx, r_idx, v_idx,eval_mode=False):
#         head = self.emb_entity[u_idx]
#         tail = self.emb_entity[v_idx]
#         r_rot = self.relation_rot[r_idx]
#         r_trans = self.relation_trans[r_idx]
#         return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]

#     def cdist(self,h,t):
#         l = h.repeat_interleave(t.shape[1],dim=0)
#         r = t.view(-1,t.shape[-1])
#         cdist = pdist(l, r).view(h.shape[0], t.shape[1]) 
#         return cdist



