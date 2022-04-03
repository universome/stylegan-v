# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import random
import numpy as np
import torch
import torch.nn.functional as F
from src.torch_utils import training_stats
from src.torch_utils import misc
from src.torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, cfg, device, G_mapping, G_synthesis, D, augment_pipe=None, G_motion_encoder=None,
                 style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.G_motion_encoder = G_motion_encoder

    def run_G(self, z, c, t, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            out = self.G_synthesis(ws, t=t, c=c)
        return out, ws

    def run_D(self, img, c, t, sync):
        if self.augment_pipe is not None:
            if self.cfg.model.loss_kwargs.get('video_consistent_aug', False):
                nf, ch, h, w = img.shape
                f = self.cfg.sampling.num_frames_per_video
                n = nf // f
                img = img.view(n, f * ch, h, w) # [n, f * ch, h, w]

            img = self.augment_pipe(img) # [n, f * ch, h, w]

            if self.cfg.model.loss_kwargs.get('video_consistent_aug', False):
                img = img.view(n * f, ch, h, w) # [n * f, ch, h, w]

        with misc.ddp_sync(self.D, sync):
            outputs = self.D(img, c, t)

        return outputs

    def accumulate_gradients(self, phase, real_img, real_c, real_t, gen_z, gen_c, gen_t, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain        = (phase in ['Gmain', 'Gboth'])
        do_Dmain        = (phase in ['Dmain', 'Dboth'])
        do_Gpl          = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1          = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        real_img = real_img.view(-1, *real_img.shape[2:]) # [batch_size * num_frames, c, h, w]

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_t, sync=(sync and not do_Gpl)) # [batch_size * num_frames, c, h, w]
                D_out_gen = self.run_D(gen_img, gen_c, gen_t, sync=False) # [batch_size]
                training_stats.report('Loss/scores/fake', D_out_gen['image_logits'])
                training_stats.report('Loss/signs/fake', D_out_gen['image_logits'].sign())
                loss_Gmain = F.softplus(-D_out_gen['image_logits']) # -log(sigmoid(y))
                if 'video_logits' in D_out_gen:
                    loss_Gmain_video = F.softplus(-D_out_gen['video_logits']).mean() # -log(sigmoid(y)) # [1]
                    training_stats.report('Loss/scores/fake_video', D_out_gen['video_logits'])
                    training_stats.report('Loss/G/loss_video', loss_Gmain_video)
                else:
                    loss_Gmain_video = 0.0 # [1]
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain + loss_Gmain_video).mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], gen_t[:batch_size], sync=sync) # [batch_size * num_frames, c, h, w]
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                with torch.no_grad():
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_t, sync=False) # [batch_size * num_frames, c, h, w]
                D_out_gen = self.run_D(gen_img, gen_c, gen_t, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', D_out_gen['image_logits'])
                training_stats.report('Loss/signs/fake', D_out_gen['image_logits'].sign())
                loss_Dgen = F.softplus(D_out_gen['image_logits']) # -log(1 - sigmoid(y))

                if 'video_logits' in D_out_gen:
                    loss_Dgen_video = F.softplus(D_out_gen['video_logits']).mean() # [1]
                    training_stats.report('Loss/scores/fake_video', D_out_gen['video_logits'])
                else:
                    loss_Dgen_video = 0.0 # [1]

            with torch.autograd.profiler.record_function('Dgen_backward'):
                (loss_Dgen + loss_Dgen_video).mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                D_out_real = self.run_D(real_img_tmp, real_c, real_t, sync=sync)
                training_stats.report('Loss/scores/real', D_out_real['image_logits'])
                training_stats.report('Loss/signs/real', D_out_real['image_logits'].sign())

                loss_Dreal = 0
                loss_Dreal_dist_preds = 0
                loss_Dreal_video = 0.0 # [1]
                if do_Dmain:
                    loss_Dreal = F.softplus(-D_out_real['image_logits']) # -log(sigmoid(y))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                    if 'video_logits' in D_out_gen:
                        loss_Dreal_video = F.softplus(-D_out_real['video_logits']).mean() # [1]
                        training_stats.report('Loss/scores/real_video', D_out_real['video_logits'])
                        training_stats.report('Loss/D/loss_video', loss_Dgen_video + loss_Dreal_video)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[D_out_real['image_logits'].sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2) # [batch_size * num_frames_per_video]
                    loss_Dr1 = loss_Dr1.view(-1, len(real_img_tmp) // len(D_out_real['image_logits'])).mean(dim=1) # [batch_size]
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            dummy_video_logits = (D_out_real["video_logits"].sum() * 0.0) if "video_logits" in D_out_real else 0.0
            with torch.autograd.profiler.record_function(name + '_backward'):
                (D_out_real["image_logits"] * 0 + dummy_video_logits + loss_Dreal + loss_Dreal_video + loss_Dr1 + loss_Dreal_dist_preds).mean().mul(gain).backward()

#----------------------------------------------------------------------------
