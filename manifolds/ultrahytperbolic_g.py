"""Hyperboloid manifold."""

import torch

from manifolds.base import Manifold
from utils.math_utils import arcosh, cosh, sinh 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from geoopt import Lorenz, Sphere

import math

class PseudoHyperboloid(Manifold):
    
    def __init__(self,space_dim=15, time_dim=1, beta=-1):
        super(PseudoHyperboloid, self).__init__()
        self.name = 'PseudoHyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6
        self.dim = space_dim + time_dim
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.dim = space_dim + time_dim
        self.beta = nn.Parameter(torch.Tensor(1))
        nn.init.constant(self.beta, beta)
        
    def _check_point_on_manifold(self, x, rtol=1e-08, atol=1e-04):
        inner = self.inner(x, x)
        ok = torch.allclose(inner, inner.new((1,)).fill_(self.beta.item()), atol=atol, rtol=rtol)
        if not ok:
            return False
        return True

    def _check_vector_on_tangent(self, x, u, rtol=1e-08, atol=1e-04):
        inner = self.inner(x, u)
        ok = torch.allclose(inner, inner.new_zeros((1,)), atol=atol, rtol=rtol)
        if not ok:
            return False
        return True

    def _check_vector_on_tangent0(self, x, atol=1e-5, rtol=1e-5):
        origin = x.clone()
        origin[:,:] = 0
        origin[:,0] = abs(self.beta)**0.5
        inner = self.inner(origin, x)
        ok = torch.allclose(inner, inner.new_zeros((1,)), atol=atol, rtol=rtol)
        if not ok:
            return False
        return True
    
    def inner(self, x, y, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        z = x * y
        res = torch.clamp(z[:,time_dim:].sum(dim=-1) - z[:, 0:time_dim].sum(dim=-1), max=self.max_norm)
        return res

    def sqdist(self, x, y, c, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        self.beta = self.beta.cuda().to(y.get_device())
        inner = self.inner(x, y, time_dim=time_dim)
        epsilon = 0.000001
        K = inner/self.beta.abs()

        c1 = K < -1.0 - epsilon
        c2 = (K < -1.0 + epsilon) & (~c1)
        c3 = (~(K < -1.0 + epsilon)) & (K < 1.0)
        c4 = K >= 1.0

        other = (~c1) & (~c2) & (~c3) & (~c4)
        dist2 = x[:,0].clone()
        assert dist2[other].shape[0] == 0
        
        device = y.get_device()
        if True in c1:
            # print('hyperbolic_like')
            dist2[c1] = (abs(self.beta)**0.5) * torch.clamp(torch.acosh(inner[c1]/self.beta),min=self.min_norm, max=self.max_norm)
        if True in c2:
            # print('Euclidean_like', K[c2].max().item(), K[c2].min().item())
            dist2[c2] = 0
        if True in c3:
            # print('shperical_like')
            dist2[c3] = (abs(self.beta)**0.5) * torch.clamp(torch.acos(inner[c3]/self.beta),min=self.min_norm, max=self.max_norm)
        if True in c4:
            # print('positive_like')
            if device != -1:
                dist2[c4] = (abs(self.beta)**0.5) * torch.clamp((torch.Tensor([math.pi]).cuda().to(device)/2 + inner[c4]/abs(self.beta)),min=self.min_norm, max=self.max_norm)
            else:
                dist2[c4] = (abs(self.beta)**0.5) * torch.clamp((torch.Tensor([math.pi])/2 + inner[c4]/abs(self.beta)),min=self.min_norm, max=self.max_norm)
        return torch.clamp(dist2**2, max=50.0)

    def expmap(self, x, v, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        epsilon = 0.000000000001
        n = v.shape[0]
        d = v.shape[1]
        inner = self.inner(v, v, time_dim=time_dim)
        norm_product = (torch.abs(inner)).sqrt()
        norm_product = torch.clamp(norm_product, max=self.max_norm).view(norm_product.size(0),-1)
        space_like = inner < -epsilon
        time_like = inner > epsilon
        null_geodesic = (~space_like) & (~time_like)

        other = (~time_like) & (~space_like) & (~null_geodesic)

        U = v.clone()
        assert U[other].shape[0] == 0

        abs_beta = 1/(abs(self.beta) ** 0.5)
        if True in time_like:
            # print('hyperbolic_like')
            beta_product = torch.clamp(abs_beta*norm_product[time_like],max=self.max_norm)
            U[time_like,:] = x[time_like,:]*torch.clamp(torch.cosh(beta_product),max=self.max_norm) +  torch.clamp(v[time_like,:]* torch.sinh(beta_product), max=self.max_norm)/(beta_product)
            assert not torch.isnan(U[time_like]).any()
        if True in space_like:
            # print('spherical_like')
            beta_product = torch.clamp(abs_beta*norm_product[space_like],max=self.max_norm)
            U[space_like,:] = x[space_like,:]*torch.clamp(torch.cos(beta_product),max=self.max_norm) +  torch.clamp(v[space_like,:]* torch.sin(beta_product), max=self.max_norm)/(beta_product)
            assert not torch.isnan(  torch.sinh(beta_product) ).any()
        if True in null_geodesic:
            # print('null_like')
            U[null_geodesic,:] = torch.clamp(x[null_geodesic,:] + v[null_geodesic,:], max=self.max_norm)
            assert not torch.isnan(v[null_geodesic,:] ).any()
        assert not torch.isnan(U).any()
        if self.time_dim == 1:
            U[:,0] = torch.abs(U[:,0])
        assert not torch.isnan(self.projx(U)).any()
        return self.projx(U)

    def expmap0(self,v, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        origin = v.clone()
        origin[:,:] = 0
        origin[:,0] = abs(self.beta)**0.5
        return self.expmap(origin, v, time_dim=time_dim)

    def logmap_n(self, x, y, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        self.beta = self.beta.cuda().to(y.get_device())

        n = y.shape[0]
        d = y.shape[1]

        inner_positive = self.inner(x,y, time_dim=time_dim)
        inner_positive = torch.clamp(inner_positive, max=self.max_norm)
        abs_beta = abs(self.beta)+1e-8
        epsilon = 0.000001  
        time_like_positive = inner_positive/abs_beta < -1 - epsilon
        null_geodesic_positive = (inner_positive/abs_beta>= -1 - epsilon) & (inner_positive/abs_beta<= -1 + epsilon)
        space_like_positive = (inner_positive/abs_beta > -1 + epsilon) & (inner_positive/abs_beta < 1)

        other = (~time_like_positive) & (~null_geodesic_positive) & (~space_like_positive)
                
        U = y.clone()
        assert U[other].shape[0] == 0
        beta_product_positive = (inner_positive/self.beta).view(inner_positive.size(0), -1)
        assert not torch.isnan(beta_product_positive).any()
        abs_da = torch.clamp((beta_product_positive**2 - 1).abs(), min=self.min_norm)
        sqrt_minus_positive = (abs_da** 0.5).view(beta_product_positive.size(0), -1)
        if True in space_like_positive:
            # print('spherical_like')
            up = torch.clamp(torch.acos(beta_product_positive[space_like_positive]), min=self.min_norm, max=self.max_norm)
            low = torch.clamp(sqrt_minus_positive[space_like_positive], min=self.min_norm, max=self.max_norm)
            U[space_like_positive,:] = ((up/low).repeat(1,d))* torch.clamp((y[space_like_positive,:]-x[space_like_positive,:]*beta_product_positive[space_like_positive].repeat(1,d)),max=self.max_norm)
            assert not torch.isnan(U[space_like_positive,:]).any()
        if True in time_like_positive:
            # print('hyperbolic_like')
            up = torch.acosh(torch.clamp(beta_product_positive[time_like_positive], min=self.min_norm, max=self.max_norm))
            low = torch.clamp(sqrt_minus_positive[time_like_positive], min=self.min_norm, max=self.max_norm)
            U[time_like_positive,:] = ((up/low).repeat(1,d))*torch.clamp( (y[time_like_positive,:]-x[time_like_positive,:]*beta_product_positive[time_like_positive].repeat(1,d)),max=self.max_norm)
            assert not torch.isnan(U[time_like_positive,:]).any()
        if True in null_geodesic_positive:
            # print('null_like')
            U[null_geodesic_positive,:] = y[null_geodesic_positive,:] - x[null_geodesic_positive,:]
            assert not torch.isnan(U[null_geodesic_positive,:]).any()
        assert not torch.isnan(U).any()
        return U

    def logmap(self,x,y, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        inner_positive = self.inner(x, y, time_dim=time_dim)
        epsilon = 0.000001
        positive_log_map = inner_positive < abs(self.beta) - epsilon
        negative_log_map = inner_positive >= abs(self.beta) + epsilon
        neutral = (~positive_log_map) & (~negative_log_map)
        U = y.clone()
        if True in positive_log_map:
            U[positive_log_map] = self.logmap_n(x[positive_log_map], y[positive_log_map], time_dim=time_dim)
        if True in negative_log_map:
            negative_logmap = self.logmap_n(-x[negative_log_map], y[negative_log_map], time_dim=time_dim)
            U[negative_log_map] = self.ptransp(-x[negative_log_map], x[negative_log_map], negative_logmap, time_dim=time_dim)
        U[neutral] = y[neutral] - x[neutral]
        return U

    def logmap0(self,y, time_dim=None):
        # print(y.max())
        time_dim = self.time_dim if time_dim==None else time_dim
        self.beta = self.beta.cuda().to(y.get_device())
        origin = y.clone()
        origin[:,:] = 0
        origin[:,0] = abs(self.beta)**0.5
        return self.logmap(origin, y,time_dim=time_dim)

    def psqrtbeta(self):
        return abs(self.beta)**0.5

    def projx(self, x, beta_scaling=False, time_dim=None):
        assert not torch.isnan(x).any()
        time_dim = self.time_dim if time_dim==None else time_dim
        if time_dim == self.dim:
            if beta_scaling:
                return self.psqrtbeta() * F.normalize(x)
            else:
                return F.normalize(x)
        
        Xtime = torch.clamp(F.normalize(x[:,0:time_dim]),max=self.max_norm)
        assert not torch.isnan(Xtime).any()
        Xspace = torch.clamp(x[:,time_dim:].div(self.psqrtbeta()),max=self.max_norm)
        assert not torch.isnan(Xspace*Xspace).any()
        spaceNorm = torch.clamp(torch.sum(Xspace*Xspace, dim=1, keepdim=True),max=self.max_norm)
        assert not torch.isnan(Xspace).any()
        if self.time_dim == 1:
            Xtime = torch.sqrt((spaceNorm).add(1.0)).view(-1,1)
        else:
            # print(torch.sqrt(spaceNorm.add(1.0)).max().item(), Xtime.max().item())
            Xtime = torch.clamp(torch.clamp(torch.sqrt(spaceNorm.add(1.0)),max=self.max_norm).expand_as(Xtime) * Xtime, max=self.max_norm)
            assert not torch.isnan(Xtime).any()

        if beta_scaling:
            return torch.clamp(self.psqrtbeta() * torch.cat((Xtime,Xspace),1),max=self.max_norm)
        else:
            return torch.cat((Xtime,Xspace),1)

    def proj_tan(self,x,z, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        inner_zx = self.inner(z,x,time_dim=time_dim)
        inner_xx = self.inner(x,x,time_dim=time_dim)
        res = z - (inner_zx/inner_xx).unsqueeze(1)*x
        return res

    def proj_tan0(self, z, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        self.beta = self.beta.cuda().to(z.get_device())
        origin = z.clone()
        origin[:,:] = 0
        origin[:,0] = abs(self.beta)**0.5
        return self.proj_tan(origin,z, time_dim=time_dim)
    
    def mobius_matvec(self, m, x, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        u = self.logmap0(x,time_dim=time_dim)
        mu = u @ m
        mu = self.proj_tan0(mu, time_dim=time_dim)
        return self.expmap0(mu,time_dim=self.time_dim)

    def mobius_add(self, x, y):
        assert not torch.isnan(x).any()
        assert not torch.isnan(y).any()
        u = self.logmap0(y)
        v = self.ptransp0(x,u)
        return self.expmap(x,v)

    def ptransp0(self,x,u):
        origin = x.clone()
        origin[:,:] = 0
        origin[:,0] = abs(self.beta)**0.5
        return self.ptransp(origin, x, u)

    def ptransp(self,x,y,u, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        inner_positive = self.inner(x, y)
        epsilon = 0.000001
        p = inner_positive < abs(self.beta) - epsilon
        n = inner_positive >= abs(self.beta) + epsilon
        neutral = (~p) & (~n)
        U = u.clone()
        U[neutral] = u[neutral]
        if True in p:
            U[p] = self.ptransp_n(x[p], y[p], u[p], time_dim=time_dim)
        if True in n:
            print("pt negative", inner_positive[n].min().item())
            negative_trans = self.ptransp_n(x[n], -y[n], u[n], time_dim=time_dim)
            U[n] = self.ptransp_n(-y[n], y[n], negative_trans) 
        U = self.proj_tan(y,U)
        return U
    
    def ptransp_n(self,x,y,u,time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        inner_xy = self.inner(x,y,time_dim=time_dim)
        log_xy = self.logmap_n(x, y, time_dim=time_dim) 
        inner_log_xy = self.inner(log_xy, log_xy, time_dim=time_dim)
        inner = self.inner(u, log_xy, time_dim=time_dim)
        inner_yu = self.inner(y,u,time_dim=time_dim)
        dist = torch.clamp(self.sqdist(x,y,None,time_dim=time_dim), min=self.min_norm, max=self.max_norm)
        # print(torch.max(log_xy), torch.max(inner_log_xy),torch.max(dist),torch.min(dist))
        epsilon = 0.000001
        time_like = inner_log_xy > epsilon
        space_like = inner_log_xy < -epsilon
        null_like = (~time_like) & (~space_like)
        other = (~time_like) & (~space_like) & (~null_like)
        
        U = u.clone()
        assert U[other].shape[0] == 0
        if True in time_like:
            # print('time_like')
            U[time_like,:] = torch.clamp((inner[time_like]/dist[time_like]).unsqueeze(1) * (x[time_like]*torch.sinh(dist[time_like]).unsqueeze(1) + (log_xy[time_like]/dist[time_like].unsqueeze(1))*torch.cosh(dist[time_like]).unsqueeze(1)) + (u[time_like] - inner[time_like].unsqueeze(1)*log_xy[time_like]/(dist[time_like]**2).unsqueeze(1)) , max=self.max_norm)
            assert not torch.isnan(U[time_like,:] ).any()
        if True in space_like:
            # print('space_like')
            U[space_like,:] = torch.clamp((inner[space_like]/dist[space_like]).unsqueeze(1)* (x[space_like]*torch.sin(dist[space_like]).unsqueeze(1) - (log_xy[space_like]/dist[space_like].unsqueeze(1))*torch.cos(dist[space_like]).unsqueeze(1)) + (u[space_like] + inner[space_like].unsqueeze(1)*log_xy[space_like]/(dist[space_like]**2).unsqueeze(1)) , max=self.max_norm)
            assert not torch.isnan(U[space_like,:] ).any()
        if True in null_like:
            # print('null_like')
            U[null_like,:] = inner[null_like].unsqueeze(1)*(x[null_like]+log_xy[null_like]/2) + u[null_like]
            assert not torch.isnan(U[null_like,:] ).any()
        return U

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.projx(x + u)


