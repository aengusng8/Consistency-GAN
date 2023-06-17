import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam, Adam

from cm import dist_util, logger
from cm.fp16_util import MixedPrecisionTrainer
from cm.nn import update_ema
from cm.resample import LossAwareSampler, UniformSampler
from cm.train_util import (
    TrainLoop,
    find_resume_checkpoint,
    log_loss_dict,
    get_blob_logdir,
)

from cm.fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)
import numpy as np
import wandb


class ConsistencyGANTrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        D,
        target_G,
        teacher_G,
        teacher_diffusion,
        training_mode,
        ema_scale_fn,
        total_training_steps,
        lazy_reg,
        r1_gamma,
        grad_clip,
        adver_focus_proportion,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.training_mode = training_mode
        self.ema_scale_fn = ema_scale_fn
        self.D = D
        self.target_G = target_G
        self.teacher_G = teacher_G
        self.teacher_diffusion = teacher_diffusion
        self.total_training_steps = total_training_steps
        self.lazy_reg = lazy_reg
        self.r1_gamma = r1_gamma
        self.grad_clip = grad_clip
        self.adver_focus_proportion = adver_focus_proportion

        if target_G:
            self._load_and_sync_target_parameters()
            self.target_G.requires_grad_(False)
            self.target_G.train()

            self.target_G_param_groups_and_shapes = get_param_groups_and_shapes(
                self.target_G.named_parameters()
            )
            # self.target_G_master_params = make_master_params(
            #     self.target_G_param_groups_and_shapes
            # )
            self.target_G_master_params = list(self.target_G.parameters())

        if teacher_G:
            self._load_and_sync_teacher_parameters()
            self.teacher_G.requires_grad_(False)
            self.teacher_G.eval()

        self.global_step = self.step

        # Generator
        self.G = self.ddp_model
        self.mp_trainerG = self.mp_trainer
        self.optimizerG = self.opt

        # Discriminator
        # TODO: create MixedPrecisionTrainer (EMA and resume, if needed) for D
        # self.D_mp_trainer = MixedPrecisionTrainer(
        #     model=self.D,
        #     use_fp16=self.use_fp16,
        #     fp16_scale_growth=self.fp16_scale_growth,
        # )
        self.optimizerD = Adam(
            self.D.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        num_epoch = self.total_training_steps // self.microbatch
        self.schedulerD = th.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizerD, num_epoch, eta_min=1e-5
        )

        self.setup_the_loss_functions()

    def setup_the_loss_functions(self):
        # 1. Consistency Generator Loss
        if self.training_mode == "consistency_gan_distillation":
            self.compute_consistency_generator_loss = functools.partial(
                self.diffusion.consistency_generator_loss,
                self.ddp_model,
                target_G=self.target_G,
                teacher_G=self.teacher_G,
                teacher_diffusion=self.teacher_diffusion,
            )
        elif self.training_mode == "consistency_gan_training":
            self.compute_consistency_generator_loss = functools.partial(
                self.diffusion.consistency_generator_loss,
                self.ddp_model,
                target_G=self.target_G,
            )
        else:
            raise ValueError(f"Unknown training mode {self.training_mode}")

        # TODO: put it in the config
        _, max_num_scale = self.ema_scale_fn(self.total_training_steps)

        # 2. Adversarial Generator Loss
        self.compute_adversarial_generator_loss = functools.partial(
            self.diffusion.adversarial_generator_loss,
            adver_focus_proportion=self.adver_focus_proportion,
            max_num_scale=max_num_scale,
        )

        # 3. Adversarial Discriminator Loss
        self.compute_adversarial_discriminator_loss = functools.partial(
            self.diffusion.adversarial_discriminator_loss,
            adver_focus_proportion=self.adver_focus_proportion,
            max_num_scale=max_num_scale,
        )

    def _load_and_sync_target_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            target_name = name.replace("model", "target_G")
            resume_target_checkpoint = os.path.join(path, target_name)
            if bf.exists(resume_target_checkpoint) and dist.get_rank() == 0:
                logger.log(
                    "loading model from checkpoint: {resume_target_checkpoint}..."
                )
                self.target_G.load_state_dict(
                    dist_util.load_state_dict(
                        resume_target_checkpoint, map_location=dist_util.dev()
                    ),
                )

        dist_util.sync_params(self.target_G.parameters())
        dist_util.sync_params(self.target_G.buffers())

    def _load_and_sync_teacher_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            teacher_name = name.replace("model", "teacher_G")
            resume_teacher_checkpoint = os.path.join(path, teacher_name)

            if bf.exists(resume_teacher_checkpoint) and dist.get_rank() == 0:
                logger.log(
                    "loading model from checkpoint: {resume_teacher_checkpoint}..."
                )
                self.teacher_G.load_state_dict(
                    dist_util.load_state_dict(
                        resume_teacher_checkpoint, map_location=dist_util.dev()
                    ),
                )

        dist_util.sync_params(self.teacher_G.parameters())
        dist_util.sync_params(self.teacher_G.buffers())

    def run_loop(self):
        saved = False
        while (
            not self.lr_anneal_steps
            or self.step < self.lr_anneal_steps
            or self.global_step < self.total_training_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            saved = False
            if (
                self.global_step
                and self.save_interval != -1
                and self.global_step % self.save_interval == 0
            ):
                self.save()
                saved = True
                th.cuda.empty_cache()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            if (self.global_step <= 1) or (self.global_step % self.log_interval == 0):
                logger.dumpkvs()

        # Save the last checkpoint if it wasn't already saved.
        if not saved:
            self.save()

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            update_ema(
                self.target_G_master_params,
                self.mp_trainerG.master_params,
                rate=target_ema,
            )
            # master_params_to_model_params(
            #     self.target_G_param_groups_and_shapes,
            #     self.target_G_master_params,
            # )

    def run_step(self, batch, cond):
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            ema, num_scales = self.ema_scale_fn(self.global_step)
            wandb.log({"num_scales": num_scales}, step=self.global_step)

            # Start updating discriminator and generator
            losses = {}

            # 1. Consistency Generator Loss
            # 1.1 Compute the G's loss
            if last_batch or not self.use_ddp:
                consis_G_loss = self.compute_consistency_generator_loss(
                    x_start=micro,
                    num_scales=num_scales,
                    G_kwargs=micro_cond,
                )
            else:
                with self.ddp_model.no_sync():
                    consis_G_loss = self.compute_consistency_generator_loss(
                        x_start=micro,
                        num_scales=num_scales,
                        G_kwargs=micro_cond,
                    )

            consis_G_loss = (consis_G_loss * weights).mean()
            losses["Consistency Generator Loss"] = consis_G_loss.item()

            # 1.2 Update G's parameters
            self.mp_trainerG.zero_grad()
            self.mp_trainerG.backward(consis_G_loss)
            took_step = self.mp_trainerG.optimize(self.optimizerG)

            # 2. Adversarial Discriminator Loss
            # 2.1 Compute the D's loss
            for p in self.D.parameters():
                p.requires_grad = True
            self.D.zero_grad()

            (
                errD_real,
                grad_penalty,
                errD_fake,
            ) = self.compute_adversarial_discriminator_loss(
                D=self.D,
                G=self.G,
                G_kwargs=micro_cond,
                x_start=micro,
                num_scales=num_scales,
                dims=micro.ndim,
                lazy_reg=self.lazy_reg,
                r1_gamma=self.r1_gamma,
                global_step=self.global_step,
                distiller=None,
            )
            # errD_real *= weights
            # errD_fake *= weights
            # grad_penalty *= weights

            # 2.2 Update D's parameters
            self.optimizerD.zero_grad()
            errD_real.backward(retain_graph=True)
            # FIXME: naive gradient norm clipping
            if grad_penalty:
                grad_penalty.backward()
            errD_fake.backward()
            # if self.grad_clip:
            #     th.nn.utils.clip_grad_norm_(self.D.parameters(), self.grad_clip)
            losses["Adversarial Discriminator Loss"] = (errD_real + errD_fake).item()
            losses["Real-Discriminator Loss"] = errD_real.item()
            losses["Fake-Discriminator Loss"] = errD_fake.item()
            losses["Gradient Penalty (Discriminator)"] = grad_penalty.item() if grad_penalty else 0.0
            self.optimizerD.step()

            for p in self.D.parameters():
                p.requires_grad = False

            # 3. Adversarial Generator Loss
            # 3.1 Compute the G's loss
            if last_batch or not self.use_ddp:
                adver_G_loss = self.compute_adversarial_generator_loss(
                    D=self.D,
                    G=self.G,
                    G_kwargs=micro_cond,
                    x_start=micro,
                    num_scales=num_scales,
                    dims=micro.ndim,
                )
            else:
                with self.ddp_model.no_sync():
                    adver_G_loss = self.compute_adversarial_generator_loss(
                        D=self.D,
                        G=self.G,
                        G_kwargs=micro_cond,
                        x_start=micro,
                        num_scales=num_scales,
                        dims=micro.ndim,
                    )

            # adver_G_loss *= weights
            losses["Adversarial Generator Loss"] = adver_G_loss.item()
            # 3.2 Update G's parameters
            self.mp_trainerG.zero_grad()
            self.mp_trainerG.backward(adver_G_loss)
            _ = self.mp_trainerG.optimize(self.optimizerG)

            # NOTE, try: G_loss = consis_G_loss
            G_loss = consis_G_loss + adver_G_loss

            # Weighting the losses
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, G_loss.detach())

            if took_step:
                self._update_ema()
                if self.target_G:
                    self._update_target_ema()
                self.step += 1
                self.global_step += 1

            # logging
            self.log_step(losses)

    def save(self):
        import blobfile as bf

        step = self.global_step

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainerG.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                if not rate:
                    filename = f"G__{step:06d}.pt"
                else:
                    filename = f"ema-G__{rate}_{step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)
                logger.log(f"Saved G state dict ({filename})")

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"optimizerG__{step:06d}.pt"),
                "wb",
            ) as f:
                th.save(self.optimizerG.state_dict(), f)

            logger.log(f"Saved optimizerG state dict (optimizerG__{step:06d}.pt)")

        if dist.get_rank() == 0:
            if self.target_G:
                filename = f"target_G__{step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(self.target_G.state_dict(), f)
                logger.log(f"Saved target G state dict ({filename})")

            if self.teacher_G and self.training_mode == "progdist":
                filename = f"teacher_G__{step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(self.teacher_G.state_dict(), f)

                logger.log(f"Saved teacher G state dict ({filename})")

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but optimizerG/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainerG.master_params)
        dist.barrier()

    def log_step(self, losses):
        # logger
        logger.logkvs(losses)
        step = self.global_step
        logger.logkv("step", step)
        samples = (step + 1) * self.global_batch
        logger.logkv("samples", samples)

        # wandb
        wandb.log(losses, step=step)
        wandb.log({"samples": samples}, step=step)
