"""Hyperboloid manifold."""

import torch

from manifolds.base import Manifold
from utils.math_utils import arcosh, cosh, sinh 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
# from geoopt import Sphere, Lorentz
# hp = Lorentz()
# sp = Sphere()



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
        self.beta.requires_grad=False
        
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

    def initialization(self,x,time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        self.beta = self.beta.cuda().to(x.get_device())
        inner = self.inner(x, x, time_dim=time_dim)
        initial_embedding = x
        epsilon = 0.00000000000
        positive = inner <= epsilon
        negative = inner > epsilon
        assert not True in negative
        initial_embedding = self.beta.abs().sqrt()*x/(inner.abs().sqrt()).unsqueeze(1)
        return self.projx(initial_embedding)


    def sqdist(self, x, y, c, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        self.beta = self.beta.cuda().to(y.get_device())
        inner = self.inner(x, y, time_dim=time_dim)
        epsilon = 0.0000001
        K = inner/self.beta.abs()

        c1 = K < -1.0 - epsilon
        c2 = (K <= -1.0 + epsilon) & (K >= -1.0 - epsilon)
        c3 = (K > -1.0 + epsilon) & (K < 1.0)
        c4 = K >= 1.0

        other = (~c1) & (~c2) & (~c3) & (~c4)
        dist2 = x[:,0].clone()
        assert dist2[other].shape[0] == 0
        
        device = y.get_device()
        if True in c1:
            # print('dist:hyperbolic_like')
            dist2[c1] = (abs(self.beta)**0.5) * torch.clamp(torch.acosh(inner[c1]/self.beta),min=self.min_norm, max=self.max_norm)
        if True in c2:
            # print('dist:Euclidean_like')
            dist2[c2] = 0
        if True in c3:
            # print(K.max().item(), self.beta.item(),'dist:shperical_like')
            dist2[c3] = (abs(self.beta)**0.5) * torch.clamp(torch.acos(inner[c3]/self.beta),min=self.min_norm, max=self.max_norm)
        if True in c4:
            if device != -1:
                dist2[c4] = (abs(self.beta)**0.5) * (torch.Tensor([math.pi]).cuda().to(device)) + self.sqdist(-x[c4], y[c4], 1)
            else:
                dist2[c4] = (abs(self.beta)**0.5) * torch.Tensor([math.pi]) + self.sqdist(-x[c4], y[c4], 1)
        return torch.clamp(dist2, max=50.0)

    def expmap(self, x, v, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        epsilon = 0.000000000001
        n = v.shape[0]
        d = v.shape[1]
        inner = self.inner(v, v, time_dim=time_dim)
        # print(inner.max(),inner.min())
        norm_product = torch.clamp(inner.abs(),min=self.min_norm).sqrt()
        norm_product = torch.clamp(norm_product, max=self.max_norm).view(norm_product.size(0),-1)

        space_like = inner < -epsilon
        time_like = inner > epsilon
        null_geodesic = (~space_like) & (~time_like)
        other = (~time_like) & (~space_like) & (~null_geodesic)
        U = v.clone()

        abs_beta = 1/(abs(self.beta) ** 0.5)
        if True in time_like:
            # print('exp:hyperbolic_like')
            beta_product = torch.clamp(abs_beta*norm_product[time_like],max=self.max_norm)
            U[time_like,:] = x[time_like,:]*torch.clamp(torch.cosh(beta_product),max=self.max_norm) +  torch.clamp( torch.clamp(v[time_like,:]*torch.sinh(beta_product), max=self.max_norm)/beta_product,  max=self.max_norm)
            assert not torch.isnan( U[time_like,:]  ).any()
        if True in space_like:
            # print('exp:spherical_like')
            beta_product = torch.clamp(abs_beta*norm_product[space_like],max=self.max_norm)
            U[space_like,:] = x[space_like,:]*torch.clamp(torch.cos(beta_product),max=self.max_norm) +  torch.clamp(torch.clamp(v[space_like,:]*torch.sin(beta_product), max=self.max_norm)/beta_product,  max=self.max_norm)
            assert not torch.isnan(  U[space_like,:] ).any()
            # U[space_like,:] = sp.expmap(x[space_like,:], v[space_like,:])
        if True in null_geodesic:
            # print('exp:null_like')
            U[null_geodesic,:] = torch.clamp(x[null_geodesic,:] + v[null_geodesic,:], max=self.max_norm)
            assert not torch.isnan(v[null_geodesic,:] ).any()
        # assert not torch.isnan(U).any()
        # assert not torch.isnan(self.projx(U)).any()
        return self.projx(U)

    def expmap0(self,v, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        self.beta = self.beta.cuda().to(v.get_device())
        origin = v.clone()
        origin[:,:] = 0
        origin[:,0] = abs(self.beta)**0.5
        p = v[:,0]>=0
        n = v[:,0]<0
        U = v.clone()
        v[:,0] = 0
        # print(U[p].shape[0], U[n].shape[0])
        if True in p:
            U[p] = self.expmap(origin[p], v[p], time_dim=time_dim)
        if True in n:
            U[n] = self.expmap(-origin[n], v[n], time_dim=time_dim)
        return U

    def logmap(self, x, y, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        self.beta = self.beta.cuda().to(y.get_device())
        d = x.shape[1]
        n = x.shape[0]
        inner_positive = self.inner(x,y, time_dim=time_dim)
        inner_positive = torch.clamp(inner_positive, max=self.max_norm)
        abs_beta = abs(self.beta)
        epsilon = 0.00000001  
        time_like_positive = inner_positive/abs_beta < -1 - epsilon
        null_geodesic_positive = (inner_positive/abs_beta>= -1 - epsilon) & (inner_positive/abs_beta<= -1 + epsilon)
        space_like_positive = (inner_positive/abs_beta > -1 + epsilon) & (inner_positive/abs_beta < 1)
        other = (~time_like_positive) & (~null_geodesic_positive) & (~space_like_positive)
                
        U = y.clone()
        # assert U[other].shape[0] == 0
        U[other,:] = 0
        beta_product_positive = (inner_positive/self.beta).view(inner_positive.size(0), -1)
        # assert not torch.isnan(beta_product_positive).any()
        abs_da = torch.clamp((beta_product_positive**2 - 1).abs(), min=self.min_norm)
        sqrt_minus_positive = (abs_da** 0.5).view(beta_product_positive.size(0), -1)
        if True in space_like_positive:
            # print('log:spherical_like')
            up = torch.clamp(torch.acos(beta_product_positive[space_like_positive]), min=self.min_norm, max=self.max_norm)
            low = torch.clamp(sqrt_minus_positive[space_like_positive], min=self.min_norm, max=self.max_norm)
            U[space_like_positive,:] = ((up/low).repeat(1,d))* torch.clamp((y[space_like_positive,:]-x[space_like_positive,:]*beta_product_positive[space_like_positive].repeat(1,d)),max=self.max_norm)
            # assert not torch.isnan(U[space_like_positive,:]).any()
        if True in time_like_positive:
            # print('log:hyperbolic_like')
            up = torch.acosh(torch.clamp(beta_product_positive[time_like_positive], min=self.min_norm, max=self.max_norm))
            low = torch.clamp(sqrt_minus_positive[time_like_positive], min=self.min_norm, max=self.max_norm)
            U[time_like_positive,:] = ((up/low).repeat(1,d))*torch.clamp( (y[time_like_positive,:]-x[time_like_positive,:]*beta_product_positive[time_like_positive].repeat(1,d)),max=self.max_norm)
            # assert not torch.isnan(U[time_like_positive,:]).any()
        if True in null_geodesic_positive:
            # print('log:null_like')
            U[null_geodesic_positive,:] = y[null_geodesic_positive,:] - x[null_geodesic_positive,:]
            # assert not torch.isnan(U[null_geodesic_positive,:]).any()
        # assert not torch.isnan(U).any()
        return U

    # def logmap(self,x,y, time_dim=None):
    #     time_dim = self.time_dim if time_dim==None else time_dim
    #     inner_positive = self.inner(x, y, time_dim=time_dim)
    #     epsilon = 0.00000001
    #     positive_log_map = inner_positive < abs(self.beta) - epsilon
    #     negative_log_map = inner_positive >= abs(self.beta) + epsilon
    #     neutral = (~positive_log_map) & (~negative_log_map)
    #     U = y.clone()
    #     other = (~positive_log_map) & (~negative_log_map) & (~neutral)
    #     assert U[other].shape[0] == 0
    #     if True in positive_log_map:
    #         # print("log:positive")
    #         U[positive_log_map] = self.logmap_n(x[positive_log_map], y[positive_log_map], time_dim=time_dim)
    #     if True in negative_log_map:
    #         # print("log:negative")
    #         U[negative_log_map] = self.logmap_n(-x[negative_log_map], y[negative_log_map], time_dim=time_dim)
    #         # U[negative_log_map] = self.ptransp(-x[negative_log_map], x[negative_log_map], median, time_dim=time_dim)
    #     U[neutral] = y[neutral] - x[neutral]
    #     self.negative_log_map = negative_log_map
    #     return U
    

    def logmap0(self,y, time_dim=None):
        # print(y.max())
        time_dim = self.time_dim if time_dim==None else time_dim
        self.beta = self.beta.cuda().to(y.get_device())
        origin = y.clone()
        origin[:,:] = 0
        origin[:,0] = abs(self.beta)**0.5
        p = y[:,0]>=0
        n = y[:,0]<0
        U = y.clone()

        if time_dim!=self.dim:
            if True in p:
                U[p] = self.logmap(origin[p], y[p], time_dim=time_dim)
                U[p,0]= abs(self.beta)**0.5
            if True in n:
                U[n] = self.logmap(-origin[n], y[n], time_dim=time_dim)
                U[n,0]= -abs(self.beta)**0.5
        else:
            U = self.logmap(origin, y, time_dim=time_dim)
        return U

    def psqrtbeta(self):
        return abs(self.beta)**0.5

    def projx(self, x, beta_scaling=False, time_dim=None):
        # assert not torch.isnan(x).any()
        time_dim = self.time_dim if time_dim==None else time_dim
        self.beta = self.beta.cuda().to(x.get_device())
        if time_dim == self.dim:
            if beta_scaling:
                U = self.psqrtbeta() * F.normalize(x)
            else:
                U =  F.normalize(x)
                return U
        
        Xtime = torch.clamp(F.normalize(x[:,0:time_dim]),max=self.max_norm)
        Xspace = torch.clamp(x[:,time_dim:].div(self.psqrtbeta()),max=self.max_norm)
        spaceNorm = torch.clamp(torch.sum(Xspace*Xspace, dim=1, keepdim=True),max=self.max_norm)
        # assert not torch.isnan(Xspace).any()
        if self.time_dim == 1:
            Xtime = torch.sqrt((spaceNorm).add(1.0)).view(-1,1)
        else:
            Xtime = torch.clamp(torch.clamp(torch.sqrt(spaceNorm.add(1.0)),max=self.max_norm).expand_as(Xtime) * Xtime, max=self.max_norm)
            # assert not torch.isnan(Xtime).any()
        if beta_scaling:
            U =  torch.clamp(self.psqrtbeta() * torch.cat((Xtime,Xspace),1),max=self.max_norm)
        else:
            U =  torch.cat((Xtime,Xspace),1)
        return U

    def proj_tan(self,x,z, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        inner_zx = self.inner(z,x,time_dim=time_dim)
        inner_xx = self.inner(x,x,time_dim=time_dim)
        res = z - (inner_zx/inner_xx).unsqueeze(1)*x
        return res

    def proj_tan0(self, z, time_dim=None):
        # print("ss")
        time_dim = self.time_dim if time_dim==None else time_dim
        self.beta = self.beta.cuda().to(z.get_device())
        origin = z.clone()
        origin[:,:] = 0
        origin[:,0] = abs(self.beta)**0.5

        p = z[:,0]>=0
        n = z[:,0]<0

        U = z.clone()

        if time_dim!=self.dim:
            U[p] = self.proj_tan(origin[p],z[p], time_dim=time_dim)
            U[p,0]= abs(self.beta)**0.5
            U[n] = self.proj_tan(-origin[n],z[n], time_dim=time_dim)
            U[n,0] = -abs(self.beta)**0.5
        else:
            U = self.proj_tan(origin,z, time_dim=time_dim)
            U[:,0]= abs(self.beta)**0.5
        return U

    def perform_rescaling_beta(self, X, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        norm_X = X * X
        norm_Xtime = norm_X[:,0:time_dim]
        norm_Xspace = norm_X[:,time_dim:]
        res = X / torch.abs( torch.sum(norm_Xspace,dim=1, keepdim=True) - torch.sum(norm_Xtime,dim=1, keepdim=True) ).sqrt().expand_as(X) * self.psqrtbeta()
        return res
    
    def mobius_matvec(self, m, x, time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        u = self.logmap0(x,time_dim=time_dim)
        mu = F.tanh(u @ m)
        mu = self.proj_tan0(mu, time_dim=time_dim)
        mu = self.expmap0(mu,time_dim=self.time_dim)
        res = self.projx(mu)
        return self.perform_rescaling_beta(res)

    def mobius_add(self, x, b):
        # bias = self.proj_tan(x,b)
        v = self.ptransp0(x,b)
        # print(x.max().item(), v.max().item())
        res = self.expmap(x,v) 
        return self.perform_rescaling_beta(res)

    # def ptransp0(self,x,b):
        
    #     origin = x.clone()
    #     origin[:,:] = 0
    #     origin[:,0] = abs(self.beta)**0.5
    #     U[n_n] = self.ptransp(-origin[n_n], x[n_n], u)
    #     return U

    def ptransp0(self,x,b):

        b=self.projx(b)
        
        origin = x.clone()
        origin[:,:] = 0
        origin[:,0] = abs(self.beta)**0.5

        if self.time_dim!=self.dim:
            p_p = (b[:,0]>=0) & (x[:,0]>=0)
            p_n = (b[:,0]>=0) & (x[:,0]<0)
            n_p = (b[:,0]<0) & (x[:,0]>=0)
            n_n = (b[:,0]<0) & (x[:,0]<0)

            U = x.clone()
            
            if True in p_p:
                # print("pp")
                u = self.logmap(origin[p_p], b[p_p])
                U[p_p] = self.ptransp(origin[p_p], x[p_p], u)
            if True in p_n:
                # print("pn")
                u = self.logmap(origin[p_n], b[p_n])
                U[p_n] = self.ptransp(origin[p_n], -x[p_n], u)
            if True in n_p:
                # print("np")
                u = self.logmap(-origin[n_p], b[n_p])
                U[n_p] = self.ptransp(-origin[n_p], -x[n_p], u)
            if True in n_n:
                # print("nn")
                u = self.logmap(-origin[n_n], b[n_n])
                U[n_n] = self.ptransp(-origin[n_n], x[n_n], u)
        else:
            u = self.logmap(origin, b)
            U = self.ptransp(origin, x, u)

        return self.proj_tan(x, U)
    
    def ptransp(self,x,y,u,time_dim=None):
        time_dim = self.time_dim if time_dim==None else time_dim
        inner_xy = self.inner(x,y,time_dim=time_dim)
        log_xy = self.logmap(x, y, time_dim=time_dim) 
        inner_log_xy = self.inner(log_xy, log_xy, time_dim=time_dim)
        inner = self.inner(u, log_xy, time_dim=time_dim)
        inner_yu = self.inner(y,u,time_dim=time_dim)
        dist = torch.clamp(self.sqdist(x,y,None,time_dim=time_dim), min=1e-5, max=self.max_norm)
        epsilon = 0.00000001
        time_like = inner_log_xy > epsilon
        space_like = inner_log_xy < -epsilon
        null_like = (~time_like) & (~space_like)
        other = (~time_like) & (~space_like) & (~null_like)
        
        U = u.clone()
        assert U[other].shape[0] == 0
        if True in time_like:
            # print('pt.time_like')
            U[time_like,:] = torch.clamp((inner[time_like]/dist[time_like]).unsqueeze(1) * (x[time_like]*torch.sinh(dist[time_like]).unsqueeze(1) + (log_xy[time_like]/dist[time_like].unsqueeze(1))*torch.cosh(dist[time_like]).unsqueeze(1)) + (u[time_like] - inner[time_like].unsqueeze(1)*log_xy[time_like]/(dist[time_like]**2).unsqueeze(1)) , max=self.max_norm)
            # U[time_like,:] = (inner[time_like]/dist[time_like]).unsqueeze(1) * (x[time_like]*torch.sinh(dist[time_like]).unsqueeze(1) + (log_xy[time_like]/dist[time_like].unsqueeze(1))*torch.cosh(dist[time_like]).unsqueeze(1)) + (u[time_like] - inner[time_like].unsqueeze(1)*log_xy[time_like]/(dist[time_like]**2).unsqueeze(1)) 
            # U[time_like,:] = hp.transp(x[time_like], y[time_like], u[time_like])
            # U[time_like,:] = u[time_like] + (inner_yu[time_like]/(self.beta.abs()-inner_xy[time_like])).unsqueeze(1)  * (x[time_like]+y[time_like])
           

        if True in space_like:
            # print('pt.space_like')
            # a = (inner[space_like]/dist[space_like]).unsqueeze(1)
            # b = x[space_like]*torch.sin(dist[space_like]).unsqueeze(1)
            # c = (log_xy[space_like]/dist[space_like].unsqueeze(1))*torch.cos(dist[space_like]).unsqueeze(1)
            # d = (u[space_like] + inner[space_like].unsqueeze(1)*log_xy[space_like]/(dist[space_like]**2).unsqueeze(1))
            U[space_like,:] = torch.clamp((inner[space_like]/dist[space_like]).unsqueeze(1)* (x[space_like]*torch.sin(dist[space_like]).unsqueeze(1) - (log_xy[space_like]/dist[space_like].unsqueeze(1))*torch.cos(dist[space_like]).unsqueeze(1)) + (u[space_like] + inner[space_like].unsqueeze(1)*log_xy[space_like]/(dist[space_like]**2).unsqueeze(1)), max=self.max_norm)
            # U[space_like,:] = sp.transp(x[space_like], y[space_like], u[space_like])
            # # print("space:",torch.max(U[space_like,:]), (inner[space_like]/dist[space_like]).max(), torch.sin(dist[space_like]).max(), (log_xy[space_like]/dist[space_like]).max(), torch.cos(dist[space_like]).max(), (log_xy[space_like]/(dist[space_like]**2)).max())
            # print("space:",U[space_like,:].max(), a.max(),b.max(),c.max(),d.max())
            # U[space_like,:] = u[space_like] - (inner_yu[space_like]/(-self.beta.abs()+inner_xy[space_like])).unsqueeze(1)  * (x[space_like]+y[space_like])
            # assert not torch.isnan(U[space_like,:] ).any()
            # print("space",U[space_like,:].max(),)
            assert not torch.isnan(U[space_like,:] ).any()
            
        if True in null_like:
            # print('null_like')
            # print('pt.null_like')
            U[null_like,:] = torch.clamp((inner[null_like]/dist[null_like]).unsqueeze(1)*(x[null_like]+log_xy[null_like]/2) + u[null_like], max=self.max_norm)
            # U[null_like,:] = u[null_like]
            assert not torch.isnan(U[null_like,:]).any()
        assert not torch.isnan(U).any()
        return U

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.projx(x + u)


