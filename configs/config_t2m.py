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
cfg.dataset.motion_rep = "full"
cfg.dataset.hml_rep = "gprvc"  ## global pos rot6d vel contact
cfg.dataset.audio_rep = "encodec"
cfg.dataset.motion_min_length_s = 2
cfg.dataset.motion_max_length_s = 10
cfg.dataset.audio_max_length_s = None
cfg.dataset.window_size_s = None
cfg.dataset.sampling_rate = 16000
cfg.dataset.text_conditioner_name = "t5-base"
cfg.dataset.audio_padding = "longest"
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


cfg.fuser = CN()
cfg.fuser.fuse_method = [{"cross": ["audio"], "prepend": ["text"]}]
cfg.fuser.cross_attention_pos_emb = False
cfg.fuser.cross_attention_pos_emb_scale = 1.0


cfg.vqvae = CN()
cfg.vqvae.body_config = "/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/checkpoints/vqvae/vqvae_rv/vqvae_rv.yaml"
cfg.vqvae.left_hand_config = None
cfg.vqvae.right_hand_config = None


cfg.motion_generator = CN()
cfg.motion_generator.target = "core.models.generation.motion_generator.MotionMuse"
cfg.motion_generator.n_q = 3  # number of streams to model
cfg.motion_generator.var_len = True
cfg.motion_generator.quality_emb = False
cfg.motion_generator.dim = 256
cfg.motion_generator.depth = 8
cfg.motion_generator.heads = 8
cfg.motion_generator.causal = False

cfg.motion_generator.flash = True
cfg.motion_generator.custom = True
cfg.motion_generator.num_tokens = 1024
cfg.motion_generator.ff_mult = 4
cfg.motion_generator.post_emb_norm = True
cfg.motion_generator.rel_pos = False
cfg.motion_generator.flatten = False
cfg.motion_generator.proj_input = False
cfg.motion_generator.spatial = False

## Conditional
cfg.motion_generator.audio_input_dim = 128
cfg.motion_generator.text_input_dim = 768
cfg.motion_generator.film = False
cfg.motion_generator.film_skip = 1


cfg.motion_generator.no_mask_token_prob = 0.1
cfg.motion_generator.cond_dropout = 0.4
cfg.motion_generator.critic_loss_weight = 0.0
##dropouts
cfg.motion_generator.attn_dropout = 0.0
cfg.motion_generator.emb_dropout = 0.1


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()
