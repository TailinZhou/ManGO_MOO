import torch
import numpy as np

from .utils import sample_v, log_normal, sample_vp_truncated_q
import torch.nn.functional as F

class VariancePreservingSDE(torch.nn.Module):
    """
    Implementation of the variance preserving VP-SDE proposed by Song et al. 2021
    See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    """ 
    def __init__(self, beta_min=0.0001, beta_max=0.02, T=1.0, t_epsilon=0.001): #20
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max #控制噪声幅度
        self.T = T
        self.t_epsilon = t_epsilon

    def beta(self, t):
        return self.beta_min + (self.beta_max-self.beta_min)*t

    def mean_weight(self, t): #= sqrt(alpha_bar(t))
        return torch.exp(-0.25 * t**2 * (self.beta_max-self.beta_min) - 0.5 * t * self.beta_min)

    def var(self, t): #=  1- alpha_bar(t) 
        return 1. - torch.exp(-0.5 * t**2 * (self.beta_max-self.beta_min) - t * self.beta_min)

    def f(self, t, x):
        return - 0.5 * self.beta(t) * x

    def g(self, t, x):
        beta_t = self.beta(t)
        return torch.ones_like(x) * beta_t**0.5
    
    def sample(self, t, x0, return_noise=False):
        """
        sample xt | x0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss   
        """
        mu = self.mean_weight(t) * x0
        std = self.var(t) ** 0.5
        epsilon = torch.randn_like(x0)
        xt = epsilon * std + mu
        if not return_noise:
            return xt
        else:
            return xt, epsilon, std, self.g(t, xt)
            # return xt, (xt-x0)/std, std, self.g(t, xt)

    def sample_debiasing_t(self, shape):
        """
        non-uniform sampling of t to debias the weight std^2/g^2
        the sampling distribution is proportional to g^2/std^2 for t >= t_epsilon
        for t < t_epsilon, it's truncated
        """
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, t_epsilon=self.t_epsilon, T=self.T)


class ScorePluginReverseSDE(torch.nn.Module):
    """
    inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    f <- g a - f
    g <- g
    (time is inverted)
    """

    def __init__(self, base_sde, drift_a, T, vtype='rademacher', debias=False):
        super().__init__()
        self.base_sde = base_sde
        self.a = drift_a
        self.T = T
        self.vtype = vtype
        self.debias = debias

    # Drift
    def mu(self, t, y, ya, lmbd=0., gamma=0.):
        a = self.a(y, self.T - t.squeeze(), ya) * (1 + gamma) - gamma * self.a(y, self.T - t.squeeze(), torch.zeros_like(ya))
        return (1. - 0.5 * lmbd) * (self.base_sde.g(self.T-t, y) ** 2) *  a - \
               self.base_sde.f(self.T - t, y)

    # Diffusion
    def sigma(self, t, y, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.base_sde.g(self.T-t, y)

    @torch.enable_grad()
    def dsm(self, x, y):
        """
        denoising score matching loss
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        x_hat, target, std, g = self.base_sde.sample(t_, x, return_noise=True)
        a = self.a(x_hat, t_.squeeze(), y)

        return ((a * std + target) ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def dsm_weighted(self, x, y, w, clip=False, c_min=None, c_max=None):
        """
        denoising score matching loss
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        x_hat, target, std, g = self.base_sde.sample(t_, x, return_noise=True)

        if clip:
            c_min = c_min.repeat((x.size(0),1))
            c_max = c_max.repeat((x.size(0),1))
            c_min = c_min.cuda()
            c_max = c_max.cuda()
            x_hat = x_hat.cuda()

            x_hat = torch.clip(x_hat, min=c_min, max=c_max)

        a = self.a(x_hat, t_.squeeze(), y)
        w_mean = torch.mean(w, dim=1, keepdim=True)
        # return (w_mean * ((a * std + target) ** 2)).view(x.size(0), -1).sum(1, keepdim=False) / 2
        return (((a * std + target) ** 2)).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def elbo_random_t_slice(self, x, y_n):
        """
        estimating the ELBO of the plug-in reverse SDE by sampling t uniformly between [0, T], and by estimating
        div(mu) using the Hutchinson trace estimator
        """
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        qt = 1 / self.T
        y = self.base_sde.sample(t_, x).requires_grad_()

        a = self.base_sde.g(t_, y) * self.a(y, t_.squeeze(), y_n)
        mu = self.base_sde.g(t_, y) * a - self.base_sde.f(t_, y)

        v = sample_v(x.shape, vtype=self.vtype).to(y)

        Mu = - (
              torch.autograd.grad(mu, y, v, create_graph=self.training)[0] * v
        ).view(x.size(0), -1).sum(1, keepdim=False) / qt

        Nu = - (a ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2 / qt
        yT = self.base_sde.sample(torch.ones_like(t_) * self.base_sde.T, x)
        lp = log_normal(yT, torch.zeros_like(yT), torch.zeros_like(yT)).view(x.size(0), -1).sum(1)

        return lp + Mu + Nu

class PluginReverseSDE(torch.nn.Module):
    """
    inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    f <- g a - f
    g <- g
    (time is inverted)
    """

    def __init__(self, base_sde, drift_a, T, vtype='rademacher', debias=False):
        super().__init__()
        self.base_sde = base_sde
        self.a = drift_a
        self.T = T
        self.vtype = vtype
        self.debias = debias

    # Drift
    def mu(self, t, y, ya, lmbd=0., gamma=0.):
        # print(gamma)
        a = self.a(y, self.T - t.squeeze(), ya) * (1 + gamma) - gamma * self.a(y, self.T - t.squeeze(), -torch.ones_like(ya))
        return (1. - 0.5 * lmbd) * self.base_sde.g(self.T-t, y) * a - \
               self.base_sde.f(self.T - t, y)
 
    # Diffusion
    def sigma(self, t, y, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.base_sde.g(self.T-t, y)

    @torch.enable_grad()
    def dsm(self, x, y):
        """
        denoising score matching loss
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        x_hat, target, std, g = self.base_sde.sample(t_, x, return_noise=True)
        a = self.a(x_hat, t_.squeeze(), y)

        return ((a * std / g + target) ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def dsm_weighted(self, x, y, w, clip=False, c_min=None, c_max=None):
        """
        denoising score matching loss
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        x_hat, target, std, g = self.base_sde.sample(t_, x, return_noise=True)

        if clip:
            c_min = c_min.repeat((x.size(0),1))
            c_max = c_max.repeat((x.size(0),1))
            c_min = c_min.cuda()
            c_max = c_max.cuda()
            x_hat = x_hat.cuda()

            x_hat = torch.clip(x_hat, min=c_min, max=c_max)

        a = self.a(x_hat, t_.squeeze(), y)
        w_mean = torch.mean(w, dim=1, keepdim=True)
        
        return (w_mean * ((a * std / g + target) ** 2)).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def elbo_random_t_slice(self, x, y_n):
        """
        estimating the ELBO of the plug-in reverse SDE by sampling t uniformly between [0, T], and by estimating
        div(mu) using the Hutchinson trace estimator
        """
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        qt = 1 / self.T
        y = self.base_sde.sample(t_, x).requires_grad_()

        a = self.a(y, t_.squeeze(), y_n)
        mu = self.base_sde.g(t_, y) * a - self.base_sde.f(t_, y)

        v = sample_v(x.shape, vtype=self.vtype).to(y)

        Mu = - (
              torch.autograd.grad(mu, y, v, create_graph=self.training)[0] * v
        ).view(x.size(0), -1).sum(1, keepdim=False) / qt

        Nu = - (a ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2 / qt
        yT = self.base_sde.sample(torch.ones_like(t_) * self.base_sde.T, x)
        lp = log_normal(yT, torch.zeros_like(yT), torch.zeros_like(yT)).view(x.size(0), -1).sum(1)

        return lp + Mu + Nu


def project_score_toguidance(score, guidance):
    # 计算余弦相似度
    cos_sim = torch.nn.functional.cosine_similarity(score, -guidance, dim=1)
    
    # 计算score在guidance上的投影
    proj = (torch.sum(score * -guidance, dim=1, keepdim=True) / 
            torch.sum(-guidance * -guidance, dim=1, keepdim=True)) * -guidance
    
    # 根据条件选择返回值
    result = torch.where(
        cos_sim.unsqueeze(1) > 0,
        proj ,
        -proj  # 如果余弦相似度小于0，返回0
    )
    
    return result


class UnconditionPluginReverseSDE(torch.nn.Module):
    """
    inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    f <- g a - f
    g <- g
    (time is inverted)
    """

    def __init__(self, base_sde, drift_a, T, vtype='rademacher', debias=False,forwardmodel=None, encoder=None, decoder=None):
        super().__init__()
        self.base_sde = base_sde
        self.a = drift_a
        self.T = T
        self.vtype = vtype
        self.debias = debias
        self.forwardmodel = forwardmodel
        self.encoder = encoder
        self.decoder = decoder

 

    # def mu(self, t, y, ya, lmbd=0., gamma=0., guidance_bool=False, guidance_scals1=0, guidance_scals2=1, return_guidance=False):
    #     # print(gamma)
    #     #   ya is not used for unconditional model, but it is passed to keep the same interface as the other models
    #     # score = self.a(y, self.T - t.squeeze(), ya) 
    #     std_t = self.base_sde.var(t)** 0.5 
    #     score = self.a(y, t.squeeze(), ya)  #* std_t  
    #     if guidance_bool:
    #         # y.requires_grad_(True)
    #         # if y.grad:
    #         #     y.grad.zero_()
    #         input_size = y.size(-1) - ya.size(-1)
    #         # self.forwardmodel.train()
    #         with torch.enable_grad():
    #             self.forwardmodel.zero_grad()
    #             y = y.requires_grad_(True) 

    #             eps = score * std_t 
    #             mean_t = self.base_sde.mean_weight(t)
    #             y0 = (y+ std_t * eps)/mean_t
    #             # y_guide = self.forwardmodel(y0[:, :input_size])
    #             # y_guide = self.forwardmodel(y[:, :input_size])
                
    #             guidance_scals1 = torch.ones_like(y[:, :input_size]) * guidance_scals1
    #             guidance_scals2 = torch.ones_like(y[:, input_size:]) * guidance_scals2
    #             guidance_scals = torch.cat((guidance_scals1, guidance_scals2), dim=-1)

    #             # loss1 = F.mse_loss(y_guide, ya, reduction='none')
    #             # loss2 = F.mse_loss(y[:, input_size:], ya, reduction='none')
    #             loss2 = F.mse_loss(y0[:, input_size:], ya, reduction='none')
    #             loss  = loss2.mean(dim=-1) #loss1.mean(dim=-1) +  
    #             loss.sum().backward()
    #             guidance = y.grad.clone().detach()

    #             scale_vector = torch.ones_like(y)
    #             scale_vector[:, input_size:] = torch.mean(torch.abs(score[:, input_size:])/torch.abs(guidance[:, input_size:]),dim=0)


    #             # print('guidance:', guidance)
    #             # guidance[:, :input_size] = torch.clamp(guidance[:, :input_size], min=-0.6, max=0.6)
    #             # guidance[:, :input_size] = torch.clamp(guidance[:, :input_size], min=-0.2, max=0.2)
    #             # guidance = torch.clamp(guidance, min=-0.5, max=0.5)
    #             # print('predict noise:', a)
    #             # print('guidance:', guidance)
    #             # print('g:' , self.base_sde.g(self.T-t, y))
    #             # print('f:',  self.base_sde.f(self.T - t, y) )
    #         # self.forwardmodel.eval()
    #         # score = project_score_toguidance(score, guidance)
    #     else:
    #         guidance  = torch.zeros_like(score)
    #         scale_vector = 1.0
    #         guidance_scals = 0.0
    #     # sigma = self.base_sde.g(self.T-t, y)
    #     sigma = self.base_sde.g( t , y)
 
    #     if return_guidance:
    #         return guidance 
    #     else:
    #         # return   (sigma**2) * (score -  guidance_scals* scale_vector* guidance) -  self.base_sde.f( t, y) 
    #         return  self.base_sde.beta(t) * (score -  guidance_scals* scale_vector* guidance) - self.base_sde.f( t, y)
    
    def mu(self, t, y, ya,  guidance_bool=False, guidance_scals1=0, guidance_scals2=1, return_guidance=False, x_min_constraint=None, x_max_constraint=None):

        std_t = self.base_sde.var(t)**0.5
        mean_t = self.base_sde.mean_weight(t)  # 确保mean_t对应alpha_t
        
        # 冻结Score网络计算score
        with torch.no_grad():
            self.a.eval()
            score =  self.a(y, t.squeeze(), ya) # * std_t # 

        if guidance_bool:
            # input_size = ya.size(-1)
            input_size = y.size(-1) - ya.size(-1)
            with torch.enable_grad():
                y = y.requires_grad_(True)
                if y.grad is not None:
                    y.grad.zero_()
                # 正确预测x0
                y0 = (y + std_t**2 * score) / mean_t
                # 计算条件损失（示例：匹配ya）
                # loss = F.mse_loss(y0[:, :input_size ], ya, reduction='none').mean(dim=-1)
                loss = F.mse_loss(y0[:, input_size:], ya, reduction='none').mean(dim=-1)
                # 约束损失计算
                if x_min_constraint is not None or x_max_constraint is not None:
                    y0_part = y0[:, :input_size]
                    lower_violation = torch.clamp(x_min_constraint - y0_part, min=0) if x_min_constraint is not None else 0
                    upper_violation = torch.clamp(y0_part - x_max_constraint, min=0) if x_max_constraint is not None else 0
                    violation = lower_violation + upper_violation
                    loss_x_constraint = (violation ** 2).mean(dim=-1)
                    loss = loss + loss_x_constraint  # 合并损失
                
                # 逐样本梯度计算
                grad_outputs = torch.ones_like(loss)
                guidance = torch.autograd.grad(loss, y, grad_outputs=grad_outputs)[0]
                # print('gradient', guidance)

                guidance_scals1 = torch.ones_like(y[:, :input_size]) * guidance_scals1
                # print('guidance_scals1', guidance_scals1)
                guidance_scals2 = torch.ones_like(y[:, input_size:]) * guidance_scals2
                guidance_scals = torch.cat((guidance_scals1, guidance_scals2), dim=-1)

                scale_vector = torch.ones_like(y)  # 初始化 scale_vector 为全1张量
                scale_vector = torch.where(
                    torch.abs(guidance) < 1e-8,
                    1.0,
                    torch.abs(score) / (torch.abs(guidance) + 1e-8)
                )

                guidance = guidance_scals* scale_vector * guidance  # 使用固定gamma缩放
                # guidance = guidance_scals * guidance  # 使用固定gamma缩放
        else:
            guidance = torch.zeros_like(score)
        
        if return_guidance:
            return guidance
        else:
            # 修正漂移项符号
            drift = self.base_sde.beta(t) * (score - guidance) - self.base_sde.f(t, y)
            return drift

    

    def sigma(self, t, y, lmbd=0.):
        return   self.base_sde.g( t , y)

    @torch.enable_grad()
    def dsm(self, x, y):
        """
        denoising score matching loss
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        x_hat, target, std, g = self.base_sde.sample(t_, x, return_noise=True)
        a = self.a(x_hat, t_.squeeze(), y)

        return ((a * std / g + target) ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def dsm_weighted(self, x, y, w, clip=False, c_min=None, c_max=None):
        """
        denoising score matching loss
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        x_hat, target, std, g = self.base_sde.sample(t_, x, return_noise=True)

        if clip:
            c_min = c_min.repeat((x.size(0),1))
            c_max = c_max.repeat((x.size(0),1))
            c_min = c_min.cuda()
            c_max = c_max.cuda()
            x_hat = x_hat.cuda()

            x_hat = torch.clip(x_hat, min=c_min, max=c_max)

        a = self.a(x_hat, t_.squeeze(), y)
        w_mean = torch.mean(w, dim=1, keepdim=True)
        
        return (w_mean * ((a * std / g + target) ** 2)).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def elbo_random_t_slice(self, x, y_n):
        """
        estimating the ELBO of the plug-in reverse SDE by sampling t uniformly between [0, T], and by estimating
        div(mu) using the Hutchinson trace estimator
        """
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        qt = 1 / self.T
        y = self.base_sde.sample(t_, x).requires_grad_()

        a = self.a(y, t_.squeeze(), y_n)
        mu = self.base_sde.g(t_, y) * a - self.base_sde.f(t_, y)

        v = sample_v(x.shape, vtype=self.vtype).to(y)

        Mu = - (
              torch.autograd.grad(mu, y, v, create_graph=self.training)[0] * v
        ).view(x.size(0), -1).sum(1, keepdim=False) / qt

        Nu = - (a ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2 / qt
        yT = self.base_sde.sample(torch.ones_like(t_) * self.base_sde.T, x)
        lp = log_normal(yT, torch.zeros_like(yT), torch.zeros_like(yT)).view(x.size(0), -1).sum(1)

        return lp + Mu + Nu
