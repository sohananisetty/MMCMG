"""
Default config
"""

import os
from glob import glob

from yacs.config import CfgNode as CN

cfg = CN()


cfg.abs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cfg.device = "cuda"

cfg.model_name = "vqvae"

cfg.pretrained_modelpath = os.path.join(
    cfg.abs_dir, f"checkpoints/{cfg.model_name}/vqvae_motion.pt"
)
cfg.output_dir = os.path.join(cfg.abs_dir, "checkpoints/")


cfg.dataset = CN()
cfg.dataset.dataset_name = "mix"
cfg.dataset.dataset_root = "/srv/hays-lab/scratch/sanisetty3/music_motion"
cfg.dataset.fps = 30
cfg.dataset.motion_rep = "full"
cfg.dataset.hml_rep = "gprvc"  ## global pos rot6d vel contact
cfg.dataset.motion_min_length_s = 2
cfg.dataset.motion_max_length_s = 10
cfg.dataset.window_size = None
cfg.dataset.sampling_rate = 16000
cfg.dataset.motion_padding = "longest"
cfg.dataset.use_motion_augmentation = False
cfg.dataset.remove_translation = False

cfg.train = CN()
cfg.train.resume = True
cfg.train.seed = 42
cfg.train.num_train_iters = 500000  #'Number of training steps
cfg.train.save_steps = 5000
cfg.train.logging_steps = 10
cfg.train.wandb_every = 100
cfg.train.evaluate_every = 5000
cfg.train.eval_bs = 20
cfg.train.train_bs = 24
cfg.train.gradient_accumulation_steps = 4
cfg.train.log_dir = os.path.join(cfg.abs_dir, f"logs/{cfg.model_name}")
cfg.train.max_grad_norm = 0.5

## optimization

cfg.train.learning_rate = 2e-4
cfg.train.weight_decay = 0.0
cfg.train.warmup_steps = 4000
cfg.train.gamma = 0.05
cfg.train.lr_scheduler_type = "cosine"


cfg.vqvae = CN()
cfg.vqvae.target = "core.models"

cfg.vqvae.nb_joints = 52
cfg.vqvae.motion_dim = 623
cfg.vqvae.dim = 512
cfg.vqvae.depth = 3
cfg.vqvae.dropout = 0.1
cfg.vqvae.down_sampling_ratio = 4
cfg.vqvae.conv_kernel_size = 5
cfg.vqvae.rearrange_output = False

cfg.vqvae.heads = 8
cfg.vqvae.codebook_dim = 768
cfg.vqvae.codebook_size = 1024
cfg.vqvae.kmeans_iters = None

cfg.vqvae.num_quantizers = 2
cfg.vqvae.quantize_dropout_prob = 0.0
cfg.vqvae.shared_codebook = False
cfg.vqvae.sample_codebook_temp = 0.2

## Loss
cfg.vqvae.commit = 1.0  # "hyper-parameter for the commitment loss"
cfg.vqvae.loss_vel = 1.0
cfg.vqvae.loss_motion = 1.0
cfg.vqvae.recons_loss = "l1_smooth"  # l1_smooth , l1 , l2
cfg.vqvae.use_geodesic_loss = False
cfg.vqvae.use_simple_loss = True


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()
