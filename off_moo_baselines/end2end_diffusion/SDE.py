import copy
import sys
import torch
import torch.nn as nn
from typing import Callable, Optional, Sequence, Union
from mo_solver.mango.design_baselines.diff.trainer import RvSDataModule
from mo_solver.mango.design_baselines.diff.nets import (
    DiffusionTest, DiffusionScore, Swish
)
from mo_solver.mango.design_baselines.diff.lib.sdes import (
    PluginReverseSDE, VariancePreservingSDE, UnconditionPluginReverseSDE
)
from mo_solver.mango.design_baselines.diff.forward import ForwardModel
import off_moo_bench as ob
import numpy as np
import copy

 
class UnConditionalPluginReverseSDE(UnconditionPluginReverseSDE):
    """Implements a unconditional SDE diffusion model."""

    def __init__(
        self,
        base_sde: VariancePreservingSDE,
        drift_a: nn.Module,
        T: torch.Tensor,
        grad_mask: torch.Tensor = None,
        forwardmodel: nn.Module = None,
        vtype: str = "rademacher",
        debias: bool = False
    ):
        """
        Args:
            base_sde: a Stochastic Differential Equation (SDE) diffusion model.
            drift_a: the drift coefficient function modeled as a neural net.
            T: maximum time step.
            grad_mask: a mask tensor indicating with dimensions of the input
                are to be optimized over.
            vtype: random vector specification for the Hutchinson trace
                estimator.
            debias: whether to use non-uniform sampling to debias the denoising
                loss.
        """
        super(UnConditionalPluginReverseSDE, self).__init__(
            base_sde=base_sde, drift_a=drift_a, T=T, vtype=vtype, debias=debias, forwardmodel=copy.deepcopy(forwardmodel)
        )
        self.grad_mask = grad_mask
        # self.forwardmodel = copy.deepcopy(forwardmodel)


    @torch.enable_grad()
    def dsm_weighted(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        is_dropout: bool = False,
        clip: bool = False,
        c_min: Optional[torch.Tensor] = None,
        c_max: Optional[torch.Tensor] = None
    ) -> torch.Tensor: 
        """
        Defines the weighted denoising score matching loss function.
        Input:
            x: a tensor of input x values.
            y: a tensor of corresponding y values.
            w: a tensor of corresponding weight values.
            clip: whether to clip the x values.
            c_min: minimum x value if clip is True.
            c_max: maximum x value if clip is True.
        Returns:
            The weighted denoising score matching loss term.
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t(
                [x.size(0)] + [1 for _ in range(x.ndim - 1)]
            )
        else:
            t_ = self.T * torch.rand(
                [x.size(0)] + [1 for _ in range(x.ndim - 1)],
                device=x.device,
                dtype=x.dtype
            )
        # noising x
        mean_weight = self.base_sde.mean_weight(t_)
        x_hat, target, std, g = self.base_sde.sample(t_, x, return_noise=True)
        x_hat = x_hat.to(torch.float32).cuda()
        # compute score function with a std scaling
        score = self.a(x_hat, t_.squeeze(), y) * std
        x_denoise = (x_hat + (score * std))/mean_weight 

        #Auxiliary loss as a tool to see the fidelity of the diffusion model (not join in the training)
        input_size = self.a.input_dim  
        y_gen = x_denoise[:,input_size:]
        # y_pre = self.forwardmodel(x_denoise[:,:input_size ])
        # guideloss1 = torch.nn.functional.mse_loss(y_pre, y, reduction='none')
        guideloss2 = torch.nn.functional.mse_loss(y_gen, y, reduction='none')
        guideloss = torch.sum( guideloss2 , dim=-1) #+guideloss1
 
        #compute noise
        w_guide =  w.reshape(-1,1) 
        eps = score * std 
 
        #compute loss
        loss = (( w_guide) * ((eps + target) ** 2)).view(x.size(0), -1) #w+ / g
        score_loss = torch.sum(loss, dim=-1) / 2.
        if is_dropout:
            return score_loss, torch.zeros_like(guideloss)
        else:
            return score_loss,  guideloss

    @torch.enable_grad()
    def dsm(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Defines the unweighted denoising score matching loss function.
        Input:
            x: a tensor of input x values.
            y: a tensor of corresponding y values.
        Returns:
            The unweighted denoising score matching loss term.
        """
        return self.dsm_weighted(x, y, w=1.0)


 
class DiffusionMOO(DiffusionTest):
    def __init__(
        self,
        taskname: str,
        task: ob.task.Task,
        grad_mask: Union[np.ndarray, torch.Tensor]=None,
        hidden_size: int = 1024,
        learning_rate: float = 1e-3,
        forwardmodel: nn.Module = None,
        beta_min: float =0.0001,
        beta_max: float = 0.005,
        simple_clip: bool = False,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        dropout_p: float = 0.0,
        activation_fn: Callable = Swish(),
        T0: float = 1.0,
        debias: bool = False,
        vtype: str = "rademacher",
        augment: bool = False,
        condition_training=True,
    ):
        """
        Args:
            taskname: the name of the offline model-based optimization (MBO)
                task.
            task: an offline model-based optimization (MBO) task.
            grad_mask: a mask tensor indicating with dimensions of the input
                are to be optimized over.
            hidden_size: hidden size of the model. Default 1024.
            learning_rate: learning rate. Default 1e-3.
            beta_min: beta_min parameter for forward SDE model training.
            beta_max: beta_max parameter for forward SDE model training.
            activation_fn: activation function. Default Swish() function.
            T0: maximum time step.
            debias: whether to use non-uniform sampling to debias the denoising
                loss.
            vtype: random vector specification for the Hutchinson trace
                estimator.
        """
        super(DiffusionMOO, self).__init__(
            taskname=taskname,
            task=task,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            beta_min=beta_min,
            beta_max=beta_max,
            dropout_p=dropout_p,
            simple_clip=simple_clip,
            clip_min=clip_min,
            clip_max=clip_max,
            activation_fn=activation_fn,
            T0=T0,
            debias=debias,
            vtype=vtype,
            augment=augment,
            condition_training=condition_training,
        )
        if grad_mask is not None:
            self.grad_mask = grad_mask
            if isinstance(self.grad_mask, np.ndarray):
                self.grad_mask = torch.from_numpy(self.grad_mask)
            self.grad_mask = self.grad_mask.to(self.device)
        else:
            self.grad_mask = None

        self.forwardmodel = forwardmodel
        if condition_training:
            print('condition_training')
            self.gen_sde = ConditionalPluginReverseSDE(
                self.inf_sde,
                self.drift_q,
                self.T,
                grad_mask=self.grad_mask,
                vtype=self.vtype,
                debias=self.debias,
                forwardmodel=self.forwardmodel,
            )
        else:
            print(f'uncondition_training', 'T:', {self.T0})
            self.gen_sde = UnConditionalPluginReverseSDE(
                self.inf_sde,
                self.drift_q,
                self.T,
                grad_mask=self.grad_mask,
                vtype=self.vtype,
                debias=self.debias,
                forwardmodel=self.forwardmodel,
            )
 
class ConditionalPluginReverseSDE(PluginReverseSDE):
    """Implements a conditional SDE diffusion model."""

    def __init__(
        self,
        base_sde: VariancePreservingSDE,
        drift_a: nn.Module,
        T: torch.Tensor,
        grad_mask: torch.Tensor = None,
        forwardmodel: nn.Module = None,
        vtype: str = "rademacher",
        debias: bool = False
    ):
        """
        Args:
            base_sde: a Stochastic Differential Equation (SDE) diffusion model.
            drift_a: the drift coefficient function modeled as a neural net.
            T: maximum time step.
            grad_mask: a mask tensor indicating with dimensions of the input
                are to be optimized over.
            vtype: random vector specification for the Hutchinson trace
                estimator.
            debias: whether to use non-uniform sampling to debias the denoising
                loss.
        """
        super(ConditionalPluginReverseSDE, self).__init__(
            base_sde=base_sde, drift_a=drift_a, T=T, vtype=vtype, debias=debias
        )
        self.grad_mask = grad_mask
        self.forwardmodel = copy.deepcopy(forwardmodel)
 