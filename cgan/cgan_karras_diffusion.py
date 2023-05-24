import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from piq import LPIPS
from torchvision.transforms import RandomCrop
from . import dist_util

from ..cm.nn import mean_flat, append_dims, append_zero
from ..cm.random_util import get_generator
from ..cm.karras_diffusion import KarrasDenoiser, get_weightings


class CGANKarrasDenoiser(KarrasDenoiser):
    def __init__(self, use_adjacent_points=False, use_ode_solver=False):
        super().__init__()
        self.use_adjacent_points = use_adjacent_points
        self.use_ode_solver = use_ode_solver

    def forward_ode(
        self,
        x_start,
        t,
        dims,
        noise=None,
    ):
        if noise is None:
            noise = th.randn_like(x_start)

        x_t = x_start + noise * append_dims(t, dims)
        return x_t

    def get_t(self, indices, num_scales):
        # BUG (maybe): indices - 1
        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        return t

    def ode_solve_adjacent_point(
        self,
        x_t1,
        t1,
        t2,
        x_start,
        teacher_model,
        teacher_denoise_fn,
        dims,
    ):
        @th.no_grad()
        def heun_solver(samples, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)

            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(samples, next_t)

            next_d = (samples - denoiser) / append_dims(next_t, dims)
            samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)

            return samples

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        ode_solver = euler_solver if teacher_model is None else heun_solver
        x_t2 = ode_solver(x_t1, t1, t2, x_start).detach()

        return x_t2

    def get_two_points_on_same_trajectory(
        self,
        x_start,
        num_scales,
        dims,
        noise,
        teacher_model,
        teacher_denoise_fn,
    ):
        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t1 = self.get_t(indices, num_scales)
        x_t1 = self.forward_ode(x_start, t1, dims, noise=noise)

        if self.use_adjacent_points:
            t2 = self.get_t(indices + 1, num_scales)

            if self.use_ode_solver:
                x_t2 = self.ode_solve_adjacent_point(
                    self,
                    x_t1,
                    t1,
                    t2,
                    x_start,
                    teacher_model,
                    teacher_denoise_fn,
                    dims,
                )
            else:
                x_t2 = self.forward_ode(x_start, t2, dims, noise=noise)

        else:
            diff = th.randint(
                1, num_scales - 1, (x_start.shape[0],), device=x_start.device
            )
            indices = (indices + diff) % num_scales
            t2 = self.get_t(indices, num_scales)
            x_t2 = self.forward_ode(x_start, t2, dims, noise=noise)

        return x_t1, t1, x_t2, t2

    def consistency_gan_losses(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        target_model=None,
        teacher_model=None,
        discriminator_model=None,
        teacher_diffusion=None,
        noise=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, **model_kwargs)[1]

        if target_model:

            @th.no_grad()
            def target_denoise_fn(x, t):
                return self.denoise(target_model, x, t, **model_kwargs)[1]

        else:
            raise NotImplementedError("Must have a target model")

        if teacher_model:

            @th.no_grad()
            def teacher_denoise_fn(x, t):
                return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]

        x_t1, x_t2, t1, t2 = self.get_two_points_on_same_trajectory(
            x_start,
            num_scales,
            dims,
            noise,
            teacher_model,
            teacher_denoise_fn,
        )

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t1, t1)

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t1)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        # Generator loss
        if self.loss_norm == "l1":
            diffs = th.abs(distiller - distiller_target)
            generator_loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            generator_loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2-32":
            distiller = F.interpolate(distiller, size=32, mode="bilinear")
            distiller_target = F.interpolate(
                distiller_target,
                size=32,
                mode="bilinear",
            )
            diffs = (distiller - distiller_target) ** 2
            generator_loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                distiller = F.interpolate(distiller, size=224, mode="bilinear")
                distiller_target = F.interpolate(
                    distiller_target, size=224, mode="bilinear"
                )

            generator_loss = (
                self.lpips_loss(
                    (distiller + 1) / 2.0,
                    (distiller_target + 1) / 2.0,
                )
                * weights
            )
        else:
            raise ValueError(f"Unknown generator_loss norm {self.loss_norm}")


        # Discriminator loss
        
        return generator_loss, discriminator_loss
