import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from configs.config_vqvae import cfg, get_cfg_defaults
from core import MotionRep
from ctl.trainer_vq_simple import VQVAEMotionTrainer


def nb_joints(motion_rep):
    if motion_rep == MotionRep.FULL:
        return 52
    elif motion_rep == MotionRep.BODY:
        return 22
    elif motion_rep == MotionRep.HAND:
        return 30
    elif motion_rep == MotionRep.LEFT_HAND:
        return 15
    elif motion_rep == MotionRep.RIGHT_HAND:
        return 15


def motion_dim(hml_rep, motion_rep, remove_trans=False):
    dim = 0
    joints = nb_joints(motion_rep)

    if "g" in hml_rep:
        if remove_trans:
            dim += 2
        else:
            dim += 4
    if "p" in hml_rep:
        if motion_rep == MotionRep.BODY or motion_rep == MotionRep.FULL:
            dim += (joints - 1) * 3
        else:
            dim += (joints) * 3
    if "r" in hml_rep:
        if motion_rep == MotionRep.BODY or motion_rep == MotionRep.FULL:
            dim += (joints - 1) * 6
        else:
            dim += (joints) * 6
    if "v" in hml_rep:
        dim += joints * 3
    if "c" in hml_rep:
        dim += 4

    return dim


def main(cfg):
    trainer = VQVAEMotionTrainer(
        args=cfg,
    ).cuda()

    trainer.train(cfg.train.resume)


if __name__ == "__main__":
    nme = "vqvae_body_gpvc_1024"
    path = f"/srv/hays-lab/scratch/sanisetty3/music_motion/MMCMG/checkpoints/vqvae/{nme}/{nme}.yaml"
    cfg = get_cfg_defaults()
    print("loading config from:", path)
    cfg.merge_from_file(path)

    cfg.vqvae.nb_joints = nb_joints(MotionRep(cfg.dataset.motion_rep))
    cfg.vqvae.motion_dim = motion_dim(
        cfg.dataset.hml_rep,
        motion_rep=MotionRep(cfg.dataset.motion_rep),
        remove_trans=cfg.dataset.remove_translation,
    )

    cfg.freeze()
    print("output_dir: ", cfg.output_dir)

    main(cfg)


# accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train.py
# accelerate launch --mixed_precision=fp16 --num_processes=1 train_vqvae.py


# accelerate configuration saved at /nethome/sanisetty3/.cache/huggingface/accelerate/default_config.yaml

# conformer_512_1024_affine
# convq_256_1024_affine
# salloc -p overcap -G 2080_ti:1 --qos debug
