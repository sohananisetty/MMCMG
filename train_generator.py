import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from configs.config_t2m import get_cfg_defaults
from ctl.trainer_muse import MotionMuseTrainer

if __name__ == "__main__":
    nme = "motion_muse_body_hands_1024"
    path = f"/srv/hays-lab/scratch/sanisetty3/music_motion/MMCMG/checkpoints/{nme}/{nme}.yaml"
    cfg = get_cfg_defaults()
    print("loading config from:", path)
    cfg.merge_from_file(path)
    cfg.freeze()
    print("output_dir: ", cfg.output_dir)

    trainer = MotionMuseTrainer(
        args=cfg,
    ).cuda()

    trainer.train(cfg.train.resume)
