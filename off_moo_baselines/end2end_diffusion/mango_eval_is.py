from contextlib import contextmanager
import torch
import torch.nn as nn
import off_moo_bench as ob
from typing import Callable, Optional, Sequence, Union
from mo_solver.mango.design_baselines.diff.nets import (
    DiffusionTest, DiffusionScore, Swish)
from pathlib import Path
from mango_utliz import *
import logging
@contextmanager
def fixed_random_seed(seed):
    """
    Context manager, used to fix random seeds and restore the random state when exiting.
    """
    # Save the current random state
    random_state = torch.random.get_rng_state()
    
    # Set a fixed random seed
    torch.manual_seed(seed)
    
    try:
        yield  
    finally:
        torch.random.set_rng_state(random_state)
def _apply_clipping(x: torch.Tensor, clip_dic: dict) -> torch.Tensor:
    """Apply tensor cropping"""
    return torch.clamp(
        x,
        min=clip_dic['clip_min'].to(device=x.device, dtype=x.dtype),
        max=clip_dic['clip_max'].to(device=x.device, dtype=x.dtype)
    )

def _initialize_samples(task, num_samples, device, augment, clip_dic, seed=42) -> torch.Tensor:
    """Initialize the sample tensor"""
    x_dim = task.x.shape[-1] * (task.x.shape[-2] if task.is_discrete else 1)
    y_dim = task.y.shape[-1] if augment else 0
    with fixed_random_seed(seed):
        X0 = torch.randn(num_samples, x_dim + y_dim, device=device)
    return _apply_clipping(X0, clip_dic) if clip_dic['simple_clip'] else X0


def _inference_scaling_step(Xt, sde, t, t_next, delta, sigma, y, task, duplicated_time, guidance_bool=False):
    """The multi-branch expansion steps when handling inference-time scaling"""
    batch_size = Xt.size(0)
    feat_dim = Xt.size(-1)
    
    # Calculate the basic drift term
    with torch.enable_grad():
        Xt_ = Xt.requires_grad_(True)
        mu = sde.gen_sde.mu(t, Xt_, y, guidance_bool=guidance_bool, guidance_scals1=0, guidance_scals2=1).clone().detach()   # [B,D]
    
    #Generate candidate samples
    dX = delta * mu.unsqueeze(1)          # [B,1,D]
    noise = torch.randn(batch_size, duplicated_time, feat_dim, 
                       device=Xt.device) * (delta**0.5) * sigma.unsqueeze(1)
    candidates = Xt.unsqueeze(1) + dX + noise  # [B,K,D]
    # compute score 
    t_next_tensor = t_next.repeat_interleave(duplicated_time, dim=0).squeeze(-1)  # [B*K]
    flat_candidates = candidates.view(-1, feat_dim)  # [B*K,D]
    scores = sde.gen_sde.a(
        flat_candidates, 
        t_next_tensor,  
        flat_candidates[:, task.x.shape[-1]:]
    )
 
    scores = scores.view(batch_size, duplicated_time, feat_dim)  # [B,K,D]
    std_t_minus_1 = torch.sqrt(sde.gen_sde.base_sde.var(t_next))[0]
    # std_t_minus_1 = std_t_minus_1.unsqueeze(1).repeat(1, duplicated_time, 1)
    mean_t_minus_1 = sde.gen_sde.base_sde.mean_weight(t_next)[0]
    # mean_t_minus_1 = mean_t_minus_1.unsqueeze(1).repeat(1, duplicated_time, 1)
    X0_pred_from_candidates =   (candidates + std_t_minus_1**3 * scores)/mean_t_minus_1
 

    # Select the optimal candidate
    y_target = y.unsqueeze(1).expand(-1, duplicated_time, -1)
    y_preds = X0_pred_from_candidates[..., task.x.shape[-1]:]
    rewards = -torch.norm(y_preds - y_target, dim=-1)  # [B, K]  
    sample_indices  = torch.argmax(rewards, dim=1) #[B]

    # no difference between these two
    # rewards = torch.norm(y_preds - y_target, dim=-1)  # [B, K]
    # weights = torch.softmax(-rewards, dim=-1)  # softmax # [B, K]
    # sample_indices  = torch.multinomial(weights, num_samples=1).squeeze(-1)  # [B]
    # print(f"sample_indices: {sample_indices}")
    # return
 
    return candidates[torch.arange(batch_size), sample_indices]


@torch.no_grad()
def heun_sampler(
    task: ob.task.Task,
    sde: Union[DiffusionTest, DiffusionScore],
    gen_condition: torch.Tensor,
    num_samples: int = 256,
    num_steps: int = 1000,
    gamma: float = 1.0,
    device: torch.device = torch.device("cuda"),
    grad_mask: Optional[torch.Tensor] = None,
    seed: int = 1000,
    augment: bool = False,
    condition_training: bool = True,
    guidance: bool = False,
    clip_dic: Optional[dict] = None,
    inference_scaling_bool: bool = False,
    duplicated_time: int = 1,
) -> Sequence[torch.Tensor]:

    # Initialization configuration
    grad_mask = grad_mask.to(device) if grad_mask is not None else None
    clip_fn = lambda x: _apply_clipping(x, clip_dic) if clip_dic['simple_clip'] else x
    X0 = _initialize_samples(task, num_samples, device, augment, clip_dic, seed=seed)

    # processing generation conditions
    gen_condition = gen_condition.to(device, dtype=torch.float32)
    y = gen_condition.unsqueeze(0).expand(num_samples, -1)


    # Initialize the time steps and states
    delta = sde.gen_sde.T.item() / num_steps
    ts = torch.linspace(1, 0, num_steps + 1, device=device) * sde.gen_sde.T.item()
    Xt = X0.clone()
    Xs = []


    # Main sampling cycle
    for i in range(num_steps):
        t = torch.full_like(Xt[:, [0]], ts[i])
        sigma = sde.gen_sde.sigma(t, Xt) #sqrt(beta(t))
        Xt = Xt.detach().clone().requires_grad_(True)
 
        # The correction steps of the Heun method
        if inference_scaling_bool and i% 5== 0 and i < num_steps - 1:
            t_minus_1 = torch.full_like(Xt[:, [0]], ts[i+1])
            Xt = _inference_scaling_step(
                Xt, sde, t, t_minus_1, delta, sigma, y, 
                task, duplicated_time, guidance_bool=guidance
            )
        else:
            with torch.enable_grad():
                Xt_ = Xt.requires_grad_(True)
                mu = sde.gen_sde.mu(t, Xt_, y, guidance_bool=guidance, guidance_scals1=0, guidance_scals2=1)
        
            noise = torch.randn_like(Xt) * (delta**0.5) * sigma
            Xt = Xt + delta * mu + noise
        
        Xt = clip_fn(Xt)
        Xs.append(Xt.cpu())

    return Xs


@torch.no_grad()
def mango_eval(
    task: ob.task.Task,
    forwardmodel: nn.Module = None,
    inverse_model: nn.Module = None,
    ckpt_dir: Union[Path, str] = "./model/mango",
    logging_dir: Optional[Union[Path, str]] = None,
    num_samples: int = 256,
    num_steps: int = 1000,
    hidden_size: int = 2048,
    seed: int = 42,
    device: str = "auto",
    score_matching: bool = False,
    gamma: float = 1.0,
    clip_dic: dict = {'simple_clip': False, 'clip_min': 0.0, 'clip_max': 1.0},
    gen_condition=None,
    augment: bool = False,
    condition_training: bool = True,
    guidance: bool = False,
    inference_scaling_bool: bool = False,
    duplicated_time: bool = False,
) -> None:
    """
    Ours  evaluation for model-based optimization (MBO).
    Input:
        task_name: name of the design-bench MBO task.
        ckpt_dir: directory to saved checkpoints.
        logging_dir: optional directory to save logs and results to.
        num_samples: number of samples. Default 2048.
        num_steps: number of integration steps for sampling. Default 1000.
        hidden_size: hidden size of the model. Default 2048.
        seed: random seed. Default 42.
        device: device. Default CPU.
        score_matching: whether to perform score matching. Default False.
        gamma: drift parameter. Default 1.0.
    Returns:
        None.
    """
    device = get_device(device)
    if task.is_discrete:
        task.map_to_logits()
    if gen_condition is None:
        gen_condition = torch.tensor(torch.zeros_like(torch.tensor(task.problem.ideal_point))).to(device)
    elif gen_condition.device != device:
        gen_condition = gen_condition.to(device)


    inverse_model = inverse_model.to(device)
    inverse_model.eval()
    if forwardmodel is not None:
        surrogate = forwardmodel
        surrogate.eval()

    designs, preds, scores = [], [], []
    grad_mask = None
    if hasattr(task.dataset, "grad_mask"):
        grad_mask = torch.from_numpy(task.dataset.grad_mask).to(device)
    
    diffusion = heun_sampler(
        task=task,
        sde=inverse_model,
        gen_condition=gen_condition,
        num_samples=num_samples,
        num_steps=num_steps,
        device=device,
        grad_mask=grad_mask,
        gamma=gamma,
        seed=seed,
        augment=augment,
        condition_training=condition_training,
        guidance=guidance,
        clip_dic=clip_dic,
        inference_scaling_bool=inference_scaling_bool,
        duplicated_time=duplicated_time,
    )
    # print(diffusion)
    idx = -1
    while diffusion[idx].isnan().any():
        idx -= 1
    X = diffusion[idx]
    if task.is_discrete:
        X = X.view(X.size(0), -1, task.x.shape[-1])
    designs = X.cpu().numpy()[np.newaxis, ...]
    scores = task.predict(X[:, :task.x.shape[-1]].cpu().numpy())#[np.newaxis, ...]
    # if 'DTLZ' in task.problem_name:
    #     scores = scores.ravel(order='F').reshape(scores.shape[-1], scores.shape[0])
    # preds = surrogate.forward(X[:, :task.x.shape[-1]].to(device)).cpu().numpy()[np.newaxis, ...]
    preds = [None]

    # Save optimization results.
    if logging_dir is not None:
        os.makedirs(logging_dir, exist_ok=True)
        # np.save(os.path.join(logging_dir, "solution_normalized.npy"  if task.is_normalized_x else "solution.npy" ), designs)
        # np.save(os.path.join(logging_dir, "predictions_normalized.npy" if task.is_normalized_y else "predictions.npy"), preds)
        # np.save(os.path.join(logging_dir, "scores.npy"), scores)
        logging.info(f"Saved experiment results to {logging_dir}")
    logging.info("Optimization complete.")

    solution = {'x': designs[0], 'y_surrogate': preds[0], 'y_scores':scores, 'algo': 'ManGO'}
    return solution

