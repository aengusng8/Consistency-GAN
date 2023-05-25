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

    def get_t(self, indices, num_scales, sigma_min=None):
        if sigma_min is None:
            sigma_min = self.sigma_min
        # BUG (maybe): indices - 1
        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        return t

    def ode_solve_adjacent_point(
        self,
        x_t1,
        t1,
        t2,
        x_start,
        teacher_netG,
        teacher_denoise_fn,
        dims,
    ):
        @th.no_grad()
        def heun_solver(samples, t, next_t, x0):
            x = samples
            if teacher_netG is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)

            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            if teacher_netG is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(samples, next_t)

            next_d = (samples - denoiser) / append_dims(next_t, dims)
            samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)

            return samples

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            if teacher_netG is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        ode_solver = euler_solver if teacher_netG is None else heun_solver
        x_t2 = ode_solver(x_t1, t1, t2, x_start).detach()

        return x_t2

    def get_two_points_on_same_trajectory(
        self,
        x_start,
        num_scales,
        dims,
        noise,
        teacher_netG,
        teacher_denoise_fn,
        use_adjacent_points,
        use_ode_solver,
    ):
        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t1 = self.get_t(indices, num_scales)
        x_t1 = self.forward_ode(x_start, t1, dims, noise=noise)

        if use_adjacent_points:
            t2 = self.get_t(indices + 1, num_scales)

            if use_ode_solver:
                x_t2 = self.ode_solve_adjacent_point(
                    self,
                    x_t1,
                    t1,
                    t2,
                    x_start,
                    teacher_netG,
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

        return x_t1, x_t2, t1, t2

    def consistency_generator_loss(
        self,
        netG,
        x_start,
        num_scales,
        netG_kwargs=None,
        target_netG=None,
        teacher_netG=None,
        teacher_diffusion=None,
        noise=None,
    ):
        if netG_kwargs is None:
            netG_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(netG, x, t, **netG_kwargs)[1]

        if target_netG:

            @th.no_grad()
            def target_denoise_fn(x, t):
                return self.denoise(target_netG, x, t, **netG_kwargs)[1]

        else:
            raise NotImplementedError("Must have a target netG")

        if teacher_netG:

            @th.no_grad()
            def teacher_denoise_fn(x, t):
                return teacher_diffusion.denoise(teacher_netG, x, t, **netG_kwargs)[1]

        x_t1, x_t2, t1, t2 = self.get_two_points_on_same_trajectory(
            x_start,
            num_scales,
            dims,
            noise,
            teacher_netG,
            teacher_denoise_fn,
            use_adjacent_points=self.use_adjacent_points,
            use_ode_solver=self.use_ode_solver,
        )

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t1, t1)

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t1)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        # Generator loss
        netG_loss = 0
        if "l2" in self.loss_norm.keys():
            diffs = (distiller - distiller_target) ** 2
            netG_loss += mean_flat(diffs) * self.loss_norm["l2"]

        if "l2-32" in self.loss_norm.keys():
            distiller = F.interpolate(distiller, size=32, mode="bilinear")
            distiller_target = F.interpolate(
                distiller_target,
                size=32,
                mode="bilinear",
            )
            diffs = (distiller - distiller_target) ** 2
            netG_loss += mean_flat(diffs) * self.loss_norm["l2-32"]

        if "lpips" in self.loss_norm.keys():
            if x_start.shape[-1] < 256:
                distiller = F.interpolate(distiller, size=224, mode="bilinear")
                distiller_target = F.interpolate(
                    distiller_target, size=224, mode="bilinear"
                )

            netG_loss += (
                self.lpips_loss(
                    (distiller + 1) / 2.0,
                    (distiller_target + 1) / 2.0,
                )
                * self.loss_norm["lpips"]
            )

        netG_loss *= weights

        return netG_loss  # Consistency Generator Loss

    def adversarial_generator_loss(
        self,
        netD,
        netG,
        netG_kwargs,
        x_start,
        num_scales,
        dims,
    ):
        def denoise_fn(x, t):
            return self.denoise(netG, x, t, **netG_kwargs)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )
        t1 = self.get_t(indices, num_scales)
        x_t1 = self.forward_ode(x_start, t1, dims, noise=noise)

        distiller = denoise_fn(x_t2, t2)

        noise = th.randn_like(x_start)
        t2 = self.get_t(self, indices + 1, num_scales, sigma_min=0)
        x_t2 = distiller + noise * append_dims(t2, dims)

        output = netD(x_t2, t1, x_t1.detach()).view(-1)
        errG = F.softplus(-output)
        errG = errG.mean()

        return errG # Adversarial Generator Loss

    def adversarial_discriminator_loss(
        self,
        netD,
        netG,
        netG_kwargs,
        x_start,
        num_scales,
        dims,
        lazy_reg,
        r1_gamma,
        global_step,
        distiller=None,
    ):
        # FIXME: What if denoise_fn instead of target_denoise_fn?
        def denoise_fn(x, t):
            return self.denoise(netG, x, t, **netG_kwargs)[1]

        # Train with real
        x_t1, x_t2, t1, t2 = self.get_two_points_on_same_trajectory(
            x_start,
            num_scales,
            dims,
            noise,
            use_adjacent_points=False,
            use_ode_solver=False,
        )

        # BUG: check shape of x_pos, t1, x_t1
        D_real = netD(x_t2, t1, x_t1.detach()).view(-1)

        errD_real = F.softplus(-D_real)
        errD_real = errD_real.mean()
        errD_real.backward(retain_graph=True)

        # encourages the discriminator to stay smooth and improves the convergence of GAN training
        if lazy_reg is None:
            grad_real = th.autograd.grad(
                outputs=D_real.sum(), inputs=x_t1, create_graph=True
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()

            grad_penalty = r1_gamma / 2 * grad_penalty
            grad_penalty.backward()
        else:
            if global_step % lazy_reg == 0:
                grad_real = th.autograd.grad(
                    outputs=D_real.sum(), inputs=x_t1, create_graph=True
                )[0]
                grad_penalty = (
                    grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()

                grad_penalty = r1_gamma / 2 * grad_penalty
                grad_penalty.backward()

        # Train with fake
        # FIXME: pass 'distiller' to avoid recomputing and complexity
        if distiller is None:
            indices = th.randint(
                0, num_scales - 1, (x_start.shape[0],), device=x_start.device
            )
            t1 = self.get_t(indices, num_scales)
            x_t1 = self.forward_ode(x_start, t1, dims, noise=noise)

            distiller = denoise_fn(x_t1, t1)

        noise = th.randn_like(x_start)

        # NOTE: x_t2 should clean if indices + 1 == 0, which means t2 == 0
        # BUG: check sigma_min=0 code
        t2 = self.get_t(self, indices + 1, num_scales, sigma_min=0)
        x_t2 = distiller + noise * append_dims(t2, dims)

        D_fake = netD(x_t2, t1, x_t1.detach()).view(-1)

        errD_fake = F.softplus(D_fake)
        errD_fake = errD_fake.mean()
        errD_fake.backward()

        return errD_real, grad_penalty, errD_fake
