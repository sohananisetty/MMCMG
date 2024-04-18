"""
Default config
"""

# import argparse
# import yaml
import os
from glob import glob

from yacs.config import CfgNode as CN

cfg = CN()


cfg.abs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cfg.device = "cuda"

cfg.model_name = "ttmodel"

cfg.output_dir = os.path.join(cfg.abs_dir, "checkpoints/")


cfg.dataset = CN()
cfg.dataset.dataset_name = "mix"
cfg.dataset.dataset_root = "/srv/hays-lab/scratch/sanisetty3/music_motion"
cfg.dataset.music_folder = "music"
cfg.dataset.fps = 30
cfg.dataset.down_sampling_ratio = 4

cfg.dataset.text_rep = "pooled_text_embed"
cfg.dataset.audio_rep = "none"
cfg.dataset.motion_rep = "full"
cfg.dataset.hml_rep = "gprvc"  ## global pos rot6d vel contact
cfg.dataset.motion_min_length_s = 2
cfg.dataset.motion_max_length_s = 10
cfg.dataset.window_size_s = None
cfg.dataset.text_conditioner_name = "t5-base"
cfg.dataset.motion_padding = "longest"


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

cfg.motion_encoder = CN()
cfg.motion_encoder.target = "core.models.TMR.ACTORStyleEncoder"
cfg.motion_encoder.nfeats = 263
cfg.motion_encoder.vae = True
cfg.motion_encoder.latent_dim = 256
cfg.motion_encoder.ff_size = 1024
cfg.motion_encoder.num_layers = 6
cfg.motion_encoder.num_heads = 4
cfg.motion_encoder.dropout = 0.1
cfg.motion_encoder.activation = "gelu"

cfg.text_encoder = CN()
cfg.text_encoder.target = "core.models.TMR.ACTORStyleEncoder"
cfg.text_encoder.nfeats = 768
cfg.text_encoder.vae = True
cfg.text_encoder.latent_dim = 256
cfg.text_encoder.ff_size = 1024
cfg.text_encoder.num_layers = 6
cfg.text_encoder.num_heads = 4
cfg.text_encoder.dropout = 0.1
cfg.text_encoder.activation = "gelu"

cfg.motion_decoder = CN()
cfg.motion_decoder.target = "core.models.TMR.ACTORStyleDecoder"
cfg.motion_decoder.nfeats = 263
cfg.motion_decoder.latent_dim = 256
cfg.motion_decoder.ff_size = 1024
cfg.motion_decoder.num_layers = 6
cfg.motion_decoder.num_heads = 4
cfg.motion_decoder.dropout = 0.1
cfg.motion_decoder.activation = "gelu"

cfg.tmr = CN()
cfg.tmr.target = "core.models.TMR.TMR"
cfg.tmr.temperature = 0.1
cfg.tmr.recons = 1.0
cfg.tmr.latent = 1.0e-5
cfg.tmr.kl = 1.0e-5
cfg.tmr.contrastive = 0.1
cfg.tmr.vae = True
cfg.tmr.threshold_selfsim = 0.80
cfg.tmr.threshold_selfsim_metrics = 0.95
cfg.tmr.fact = None


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()
