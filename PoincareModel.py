import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoopt import PoincareBall

from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, logmap0, project, hyp_distance

pdist = nn.PairwiseDistance(p=2)
ball = PoincareBall(c=1.0)

class RotE(torch.nn.Module):
    def __init__(self, d, dim, max_scale, max_norm, margin):
        super(RotE, self).__init__()
        self.emb_entity = nn.Parameter((1e-5)*torch.randn((len(d.entities), dim)))
        self.relation_rot= nn.Parameter(2*torch.rand((len(d.relations), dim))-1.0)
        self.relation_rot_center = nn.Parameter(torch.randn((len(d.relations), dim)))
        self.relation_trans = nn.Parameter(torch.randn((len(d.relations), dim)))

        self.margin = margin
        self.bias_head = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.bias_tail = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, u_idx, r_idx, v_idx,eval_mode=False):
        head = self.emb_entity[u_idx]
        tail = self.emb_entity[v_idx]
        r_rot = self.relation_rot[r_idx]
        r_trans = self.relation_trans[r_idx]
        r_center = self.relation_rot_center[r_idx]
        head = givens_rotations(r_rot, head+r_center) + r_trans
        dist = torch.norm( head.unsqueeze(1).repeat(1,tail.shape[1],1)-tail, dim=-1, p=2)
        neg_dist = self.margin - dist
        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]

class BuseE(torch.nn.Module):
    def __init__(self, d, dim, max_scale, max_norm, margin):
        super(BuseE, self).__init__()
        self.data_type = torch.double
        self.init_size = 0.001
        self.emb_entity = nn.Parameter(self.init_size *torch.randn((len(d.entities), dim), dtype=self.data_type))
        self.rel_diag = nn.Parameter(2*torch.rand((len(d.relations), dim), dtype=self.data_type)-1.0)
        self.relation_bias_1 = nn.Parameter(self.init_size *torch.randn((len(d.relations), dim), dtype=self.data_type))
        self.relation_bias_2 = nn.Parameter(self.init_size *torch.randn((len(d.relations), dim), dtype=self.data_type))
        
        self.margin = margin
        self.bias_head = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=self.data_type))
        self.bias_tail = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=self.data_type))
        self.sigma = torch.nn.Parameter(torch.rand(len(d.relations), dtype=self.data_type))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx):
        head = ball.expmap0(self.emb_entity[u_idx])
        tail = ball.expmap0(self.emb_entity[v_idx])
        r_bias_1 = ball.expmap0(self.relation_bias_1[r_idx])
        r_bias_2 = ball.expmap0(self.relation_bias_2[r_idx])
        r_diag = self.rel_diag[r_idx]

        head = ball.mobius_add(head, r_bias_1)
        head = givens_rotations(r_diag, head)
        head = ball.mobius_add(head, r_bias_2)
        dist_tail = busemann_distance(tail, head.unsqueeze(1).repeat(1,tail.shape[1],1)).squeeze(-1)
        dist_head = busemann_distance(head.unsqueeze(1).repeat(1,tail.shape[1],1), tail).squeeze(-1)
        sigma = F.sigmoid(self.sigma[r_idx]).unsqueeze(-1)
        dist = sigma*dist_tail + (1-sigma)*dist_head
        neg_dist =  self.margin - dist
        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]

class RotH(torch.nn.Module):
    def __init__(self, d, dim, max_scale, max_norm, margin):
        super(RotH, self).__init__()
        self.data_type = torch.double
        self.init_size = 0.001
        self.emb_entity = nn.Parameter(self.init_size *torch.randn((len(d.entities), dim), dtype=self.data_type))
        self.rel_diag = nn.Parameter(2*torch.rand((len(d.relations), dim), dtype=self.data_type)-1.0)
        self.relation_bias_1 = nn.Parameter(self.init_size *torch.randn((len(d.relations), dim), dtype=self.data_type))
        self.relation_bias_2 = nn.Parameter(self.init_size *torch.randn((len(d.relations), dim), dtype=self.data_type))
        
        self.margin = margin
        self.bias_head = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=self.data_type))
        self.bias_tail = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=self.data_type))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx):
        head = ball.expmap0(self.emb_entity[u_idx])
        tail = ball.expmap0(self.emb_entity[v_idx])
        r_bias_1 = ball.expmap0(self.relation_bias_1[r_idx])
        r_bias_2 = ball.expmap0(self.relation_bias_2[r_idx])
        r_diag = self.rel_diag[r_idx]

        head = ball.mobius_add(head, r_bias_1)
        head = givens_rotations(r_diag, head)
        head = ball.mobius_add(head, r_bias_2)
        dist = hyp_distance(tail, head.unsqueeze(1).repeat(1,tail.shape[1],1)).squeeze(-1)
        neg_dist =  self.margin - dist
        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]

class ProjH(torch.nn.Module):
    def __init__(self, d, dim, max_scale, max_norm, margin):
        super(ProjH, self).__init__()
        self.data_type = torch.double
        self.init_size = 0.001
        self.emb_entity = nn.Parameter(self.init_size *torch.randn((len(d.entities), dim), dtype=self.data_type))
        self.rel_diag = nn.Parameter(2*torch.rand((len(d.relations), dim), dtype=self.data_type)-1.0)
        self.relation_bias_1 = nn.Parameter(self.init_size *torch.randn((len(d.relations), dim), dtype=self.data_type))
        self.relation_bias_2 = nn.Parameter(self.init_size *torch.randn((len(d.relations), dim), dtype=self.data_type))
        self.rel_plane = nn.Parameter(self.init_size*torch.randn((len(d.relations), dim)))
        
        self.margin = margin
        self.bias_head = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.bias_tail = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.loss = torch.nn.BCEWithLogitsLoss()

        
    def forward(self, u_idx, r_idx, v_idx):
        head = ball.expmap0(self.emb_entity[u_idx])
        tail = ball.expmap0(self.emb_entity[v_idx])
        r_bias_1 = ball.expmap0(self.relation_bias_1[r_idx])
        r_bias_2 = ball.expmap0(self.relation_bias_2[r_idx])
        rel_plane = ball.expmap0(self.rel_plane[r_idx])
        r_diag = self.rel_diag[r_idx]

        head = ball.mobius_add(head, r_bias_1)
        head = givens_rotations(r_diag, head)
        head = ball.mobius_add(head, r_bias_2)

        head = self.project(head, rel_plane)
        tail = self.project(tail, rel_plane)

        dist = ball.dist2(head.unsqueeze(1).repeat(1,tail.shape[1],1), tail, dim=-1, keepdim=False)
        neg_dist =  self.margin - dist
        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]

    def project(self, x, rel_plane):
        return psi_t(euc_projection(psi(x),rel_plane))

def psi(x):
    norm = x.norm(dim=-1, p=2, keepdim=True)
    xt = torch.tanh(2*torch.atanh(norm))/norm
    return x*xt

def psi_t(x):
    norm = x.norm(dim=-1, p=2, keepdim=True)
    xt = torch.tanh(0.5*torch.atanh(norm))/norm
    return x*xt

def euc_projection(x, w):
    # print(w.shape,x.shape)
    if w.shape[1]!=x.shape[1]:
        w = w.unsqueeze(1).repeat(1,x.shape[1],1)
    return x - torch.sum( w * x, dim=-1, keepdim=True) * w

class MuRP(torch.nn.Module):
    """Diagonal scaling https://arxiv.org/pdf/1905.09791.pdf"""
    def __init__(self, d, dim, max_scale, max_norm, margin):
        super(MuRP, self).__init__()
        self.data_type = torch.double
        self.init_size = 0.001
        self.emb_entity = nn.Parameter(self.init_size * torch.randn((len(d.entities), dim), dtype=self.data_type))
        self.rel_diag = nn.Parameter(2*torch.rand((len(d.relations), dim), dtype=self.data_type)-1.0)
        self.relation_bias = nn.Parameter(self.init_size * torch.randn((len(d.relations), dim), dtype=self.data_type))
        
        self.margin = margin
        self.bias_head = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=self.data_type))
        self.bias_tail = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=self.data_type))
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, u_idx, r_idx, v_idx):
        head = ball.expmap0(self.emb_entity[u_idx])
        tail = ball.expmap0(self.emb_entity[v_idx])
        r_bias = ball.expmap0(self.relation_bias[r_idx])
        r_diag = self.rel_diag[r_idx]

        head = ball.mobius_add(ball.expmap0( r_diag * ball.logmap0(head) ), r_bias)
        dist = ball.dist2(head.unsqueeze(1).repeat(1,tail.shape[1],1), tail, dim=-1, keepdim=False)

        neg_dist =  self.margin - dist
        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]
