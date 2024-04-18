import itertools
import os
import random
from collections import Counter
from functools import partial
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import transformers
import wandb
from core import MotionRep
from core.datasets.conditioner import ConditionProvider
from core.datasets.vq_dataset import load_dataset, simple_collate
from core.models.loss import ReConsLoss
from core.models.utils import instantiate_from_config
from core.optimizer import get_optimizer
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_scheduler
from yacs.config import CfgNode


def exists(val):
    return val is not None


def noop(*args, **kwargs):
    pass


def cycle(dl):
    while True:
        for data in dl:
            yield data


def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def yes_or_no(question):
    answer = input(f"{question} (y/n) ")
    return answer.lower() in ("yes", "y")


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


# auto data to module keyword argument routing functions


def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return any(filter(lambda count: count > 1, counts.values()))


# main trainer class


class VQVAEMotionTrainer(nn.Module):
    def __init__(
        self,
        args: CfgNode,
    ):
        super().__init__()
        self.model_name = args.model_name

        transformers.set_seed(42)

        self.args = args
        self.vqvae_args = args.vqvae
        self.training_args = args.train
        self.dataset_args = args.dataset
        self.dataset_name = args.dataset.dataset_name
        self.num_train_steps = self.training_args.num_train_iters
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.register_buffer("steps", torch.Tensor([0]))
        self.motion_rep = MotionRep(self.dataset_args.motion_rep)
        self.hml_rep = self.dataset_args.hml_rep
        self.remove_translation = self.dataset_args.remove_translation
        print(self.vqvae_args)

        self.vqvae_model = instantiate_from_config(self.vqvae_args).to(self.device)

        total = sum(p.numel() for p in self.vqvae_model.parameters() if p.requires_grad)
        print("Total training params: %.2fM" % (total / 1e6))

        self.grad_accum_every = self.training_args.gradient_accumulation_steps

        self.loss_fnc = ReConsLoss(
            recons_loss=self.vqvae_args.recons_loss,
            use_geodesic_loss=self.vqvae_args.use_geodesic_loss,
            nb_joints=self.vqvae_args.nb_joints,
            hml_rep=self.hml_rep,
            motion_rep=self.motion_rep,
            use_simple_loss=self.vqvae_args.use_simple_loss,
            remove_translation=self.remove_translation,
        )

        self.optim = get_optimizer(
            self.vqvae_model.parameters(),
            lr=self.training_args.learning_rate,
            wd=self.training_args.weight_decay,
        )

        self.lr_scheduler = get_scheduler(
            name=self.training_args.lr_scheduler_type,
            optimizer=self.optim,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=self.num_train_steps,
        )

        self.max_grad_norm = self.training_args.max_grad_norm

        dataset_names = {
            "animation": 0.8,
            "humanml": 3.0,
            "perform": 0.6,
            "GRAB": 1.0,
            "idea400": 2.0,
            "humman": 0.5,
            "beat": 2.5,
            "game_motion": 0.8,
            "music": 0.5,
            "aist": 2.0,
            "fitness": 1.0,
            "moyo": 1.5,
            "choreomaster": 2.5,
            "dance": 1.0,
            "kungfu": 1.0,
            "EgoBody": 0.5,
            # "HAA500": 1.0,
        }

        train_ds, sampler_train, _ = load_dataset(
            dataset_names=list(dataset_names.keys()),
            dataset_args=self.dataset_args,
            split="train",
            weight_scale=list(dataset_names.values()),
        )
        test_ds, _, _ = load_dataset(
            dataset_names=list(dataset_names.keys()),
            dataset_args=self.dataset_args,
            split="test",
        )
        self.render_ds, _, _ = load_dataset(
            dataset_names=list(dataset_names.keys()),
            dataset_args=self.dataset_args,
            split="render",
        )

        # if self.is_main:
        self.print(
            f"training with training {len(train_ds)} and test dataset of  and  {len(test_ds)} samples and render of  {len(self.render_ds)}"
        )

        # dataloader

        condition_provider = ConditionProvider(
            motion_rep=MotionRep(self.dataset_args.motion_rep),
            only_motion=True,
        )

        self.dl = DataLoader(
            train_ds,
            batch_size=self.training_args.train_bs,
            sampler=sampler_train,
            shuffle=False if sampler_train else True,
            collate_fn=partial(simple_collate, conditioner=condition_provider),
        )
        self.valid_dl = DataLoader(
            test_ds,
            batch_size=self.training_args.eval_bs,
            shuffle=False,
            collate_fn=partial(simple_collate, conditioner=condition_provider),
        )
        self.render_dl = DataLoader(
            self.render_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=partial(simple_collate, conditioner=condition_provider),
        )

        self.dl_iter = cycle(self.dl)

        self.save_model_every = self.training_args.save_steps
        self.log_losses_every = self.training_args.logging_steps
        self.evaluate_every = self.training_args.evaluate_every
        self.calc_metrics_every = self.training_args.evaluate_every
        self.wandb_every = self.training_args.wandb_every

        wandb.login()
        wandb.init(project=self.model_name)

    def print(self, msg):
        # self.accelerator.print(msg)
        print(msg)

    @property
    def device(self):
        return torch.device("cuda")

    def save(self, path, loss=None):
        pkg = dict(
            model=self.vqvae_model.state_dict(),
            optim=self.optim.state_dict(),
            steps=self.steps,
            total_loss=self.best_loss if loss is None else loss,
            config=dict(self.args),
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location="cuda")
        self.vqvae_model.load(str(path))

        self.optim.load_state_dict(pkg["optim"])
        self.steps = pkg["steps"]
        self.best_loss = pkg["total_loss"]

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(
            self.vqvae_args.codebook_size, code_idx.shape[0], device=code_idx.device
        )  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    def train_step(self):
        steps = int(self.steps.item())

        self.vqvae_model = self.vqvae_model.train()

        # logs

        logs = {}

        for _ in range(self.grad_accum_every):
            batch = next(self.dl_iter)

            gt_motion = batch["motion"][0].to(self.device)

            if self.remove_translation:

                if "g" in self.dataset_args.hml_rep:
                    l = list(range(0, gt_motion.shape[-1]))
                    ohprvc = l[:1] + l[3:]
                    gt_motion = gt_motion[..., ohprvc]
            mask = batch["motion"][1].to(self.device)

            vqvae_output = self.vqvae_model(
                motion=gt_motion,
                # mask=mask,
            )

            loss_motion = self.loss_fnc(
                vqvae_output.decoded_motion, gt_motion, mask=None
            )

            loss = (
                self.vqvae_args.loss_motion * loss_motion
                + self.vqvae_args.commit * vqvae_output.commit_loss
            ) / self.grad_accum_every

            # usage = len(set(used_indices)) / self.vqvae_args.codebook_size

            # print(loss,loss.shape)

            loss.backward()
            perplexity = self.compute_perplexity(vqvae_output.indices.flatten())

            accum_log(
                logs,
                dict(
                    loss=loss.detach().cpu(),
                    loss_motion=loss_motion.detach().cpu() / self.grad_accum_every,
                    commit_loss=vqvae_output.commit_loss.detach().cpu()
                    / self.grad_accum_every,
                    perplexity=perplexity.detach().cpu() / self.grad_accum_every,
                ),
            )

        self.optim.step()
        self.lr_scheduler.step()
        self.optim.zero_grad()

        # build pretty printed losses

        losses_str = f"{steps}: vqvae model total loss: {logs['loss'].float():.3} reconstruction loss: {logs['loss_motion'].float():.3} commit_loss: {logs['commit_loss'].float():.3} codebook usage: {logs['perplexity']}"

        # log
        if steps % self.wandb_every == 0:
            for key, value in logs.items():
                wandb.log({f"train_loss/{key}": value})

            self.print(losses_str)

        # if self.is_main and not (steps % self.save_model_every) and steps > 0:
        if not (steps % self.save_model_every):
            os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
            model_path = os.path.join(
                self.output_dir, "checkpoints", f"vqvae_motion.{steps}.pt"
            )
            self.save(model_path)
            print(float(logs["loss"]), self.best_loss)

            if float(logs["loss"]) <= self.best_loss:
                model_path = os.path.join(self.output_dir, f"vqvae_motion.pt")
                self.save(model_path)
                self.best_loss = logs["loss"]

            self.print(
                f'{steps}: saving model to {str(os.path.join(self.output_dir , "checkpoints") )}'
            )

        if steps % self.evaluate_every == 0:

            self.validation_step()
            self.sample_render_hmlvec(os.path.join(self.output_dir, "samples"))

        self.steps += 1
        return logs

    def validation_step(self):
        self.vqvae_model.eval()
        val_loss_ae = {}

        self.print(f"validation start")

        cnt = 0

        with torch.no_grad():
            for batch in tqdm(
                (self.valid_dl),
                position=0,
                leave=True,
                # disable=not self.accelerator.is_main_process,
            ):
                gt_motion = batch["motion"][0].to(self.device)
                if self.remove_translation:

                    if "g" in self.dataset_args.hml_rep:
                        l = list(range(0, gt_motion.shape[-1]))
                        ohprvc = l[:1] + l[3:]
                        gt_motion = gt_motion[..., ohprvc]
                    mask = batch["motion"][1].to(self.device)

                vqvae_output = self.vqvae_model(
                    motion=gt_motion,
                    # mask=mask,
                )

                loss_motion = self.loss_fnc(
                    vqvae_output.decoded_motion, gt_motion, mask=None
                )

                loss = (
                    self.vqvae_args.loss_motion * loss_motion
                    + self.vqvae_args.commit * vqvae_output.commit_loss
                ) / self.grad_accum_every

                perplexity = self.compute_perplexity(vqvae_output.indices.flatten())

                loss_dict = {
                    "total_loss": loss.detach().cpu(),
                    "loss_motion": loss_motion.detach().cpu(),
                    "commit_loss": vqvae_output.commit_loss.detach().cpu(),
                    "perplexity": perplexity,
                }

                for key, value in loss_dict.items():
                    if key in val_loss_ae:
                        val_loss_ae[key] += value
                    else:
                        val_loss_ae[key] = value

                cnt += 1

        for key in val_loss_ae.keys():
            val_loss_ae[key] = val_loss_ae[key] / cnt

        for key, value in val_loss_ae.items():
            wandb.log({f"val_loss_/{key}": value})

        print(
            "val/rec_loss",
            val_loss_ae["loss_motion"],
        )
        print(
            f"val/total_loss ",
            val_loss_ae["total_loss"],
        )
        print(
            f"val/usage ",
            val_loss_ae["perplexity"],
        )

        self.vqvae_model.train()

    def sample_render_hmlvec(self, save_path):
        save_file = os.path.join(save_path, f"{int(self.steps.item())}")
        os.makedirs(save_file, exist_ok=True)

        assert self.render_dl.batch_size == 1, "Batch size for rendering should be 1!"

        dataset_lens = self.render_ds.cumulative_sizes
        self.vqvae_model.eval()
        print(f"render start")
        with torch.no_grad():
            for idx, batch in tqdm(
                enumerate(self.render_dl),
            ):

                gt_motion = batch["motion"][0].to(self.device)

                if self.remove_translation:

                    if "g" in self.dataset_args.hml_rep:
                        l = list(range(0, gt_motion.shape[-1]))
                        ohprvc = l[:1] + l[3:]
                        gt_motion = gt_motion[..., ohprvc]
                    mask = batch["motion"][1].to(self.device)

                name = str(batch["names"][0])

                curr_dataset_idx = np.searchsorted(dataset_lens, idx + 1)
                dset = self.render_ds.datasets[curr_dataset_idx]

                vqvae_output = self.vqvae_model(gt_motion)

                pred_motion = vqvae_output.decoded_motion.squeeze().cpu()
                gt_motion = gt_motion.squeeze().cpu()

                if self.remove_translation:

                    z = torch.zeros(
                        gt_motion.shape[:-1] + (2,),
                        dtype=gt_motion.dtype,
                        device=gt_motion.device,
                    )

                    pred_motion = torch.cat(
                        [pred_motion[..., 0:1], z, pred_motion[..., 1:]], -1
                    )
                    gt_motion = torch.cat(
                        [gt_motion[..., 0:1], z, gt_motion[..., 1:]], -1
                    )

                dset.render_hml(
                    gt_motion,
                    os.path.join(
                        save_file, os.path.basename(name).split(".")[0] + "_gt.gif"
                    ),
                    # zero_trans=True,
                )

                dset.render_hml(
                    pred_motion,
                    os.path.join(
                        save_file, os.path.basename(name).split(".")[0] + "_pred.gif"
                    ),
                    # zero_trans=True,
                )

        self.vqvae_model.train()

    def train(self, resume=False, log_fn=noop):
        self.best_loss = float("inf")
        print(self.output_dir)

        if resume:
            save_dir = self.args.output_dir
            save_path = os.path.join(save_dir, "vqvae_motion.pt")
            print("resuming from ", save_path)
            self.load(save_path)

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print("training complete")
