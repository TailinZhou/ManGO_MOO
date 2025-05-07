import os
from typing import Callable, Optional, Sequence, Union
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
import off_moo_bench as ob
from mo_solver.mango.design_baselines.diff.trainer import RvSDataModule
import logging
def mango_train(
    task: ob.task.Task,
    X: np.ndarray,
    y: np.ndarray,
    ckpt_dir: Union[Path, str] = None,
    forward_model: nn.Module = None,
    inverse_model: nn.Module = None,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    device: str = "auto",
    num_epochs: int = 100,
    lr: float = 1e-3,
    hidden_size: int = 2048,
    dropout: float = 0.0,
    score_matching: bool = False,
    data_preserved_ratio: float = 1.0,
    clip_dic: dict = {'simple_clip': False, 'clip_min': 0.0, 'clip_max': 1.0},
    debais: bool = False,
    augment: bool = False,
    condition_training: bool = True,
) -> None:
    """
    Our model training for model-based optimization (MBO).
    Input:
        task_name: name of the design-bench MBO task.
        ckpt_dir: directory to saved checkpoints.
        batch_size: batch size. Default 128.
        num_workers: number of workers. Default 0.
        seed: random seed. Default 42.
        device: device. Default CPU.
        num_epochs: number of training epochs. Default 100.
        lr: learning rate. Default 0.001.
        hidden_size: hidden size of the model. Default 2048.
        dropout: dropout probability. Default 0.0
        score_matching: whether to perform score matching. Default False.
    Returns:
        None.
    """
    if ckpt_dir is None:
        ckpt_dir = "./model/mango"
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Saving our model to {ckpt_dir}")
    if task.is_discrete:
        task.map_to_logits()
    # if task_name == os.environ["CHEMBL_TASK"]:
    #     task.map_normalize_y()
    # task = task
    # forward_model = forward_model

    if inverse_model is None:
        inverse_model = DiffusionMOO(
            taskname=task.dataset_name,
            task=task,
            # grad_mask=task.dataset.grad_mask,
            learning_rate=lr,
            dropout_p=dropout,
            hidden_size=hidden_size,
            simple_clip=clip_dic['simple_clip'],
            clip_min= clip_dic['clip_min'],
            clip_max= clip_dic['clip_max'],
            debias=debais,
            forwardmodel=forward_model,
            augment=augment,
            condition_training=condition_training
        )

 

    dm = RvSDataModule(
        task=task,
        X=X,
        y=y,
        val_frac=0.1,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        temp="90",
        data_preserved_ratio=data_preserved_ratio,
        augment=augment,
    )
    

    devices = "".join(filter(str.isdigit, device))
    devices = [int(devices)] if len(devices) > 0 else "auto"
    


    accelerator = device.split(":")[0].lower()
    accelerator = "gpu" if accelerator == "cuda" else accelerator
    # pl.seed_everything(seed)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy='dp',#'dp
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="elbo_estimator", #
                dirpath=ckpt_dir,
                filename=f"ours-{task.dataset_name}-seed{seed}-hidden{hidden_size}-score_matching{score_matching}"
            )
        ], 
        max_epochs=num_epochs,
        logger=False,
        gpus=1,
        # scheduler="cosine",
    )
    trainer.fit(inverse_model, dm)
    logging.info(f"Saved trained our diffusion model to {ckpt_dir}")

    return inverse_model
