import functools
from typing import Tuple, List, Dict

import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.torch_utils import persistence
from src.training.networks import Discriminator as ImageDiscriminator

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(nn.Module):
    """
    MoCoGAN discriminator, consisting on 2 parts: ImageDiscriminator and VideoDiscriminator
    """
    def __init__(self,
        cfg: DictConfig,
        img_channels: int,
        img_resolution: int,
        *img_discr_args,
        **img_discr_kwargs):

        super().__init__()

        self.cfg = cfg
        self.image_discr = ImageDiscriminator(
            img_resolution=img_resolution,
            img_channels=img_channels,
            cfg=OmegaConf.create({
                'sampling': {'num_frames_per_video': 1},
                'dummy_c': False,
                'fmaps': 1.0 if img_resolution >= 512 else 0.5,
                'mbstd_group_size': 4,
                'concat_res': -1,
            }),
            *img_discr_args,
            **img_discr_kwargs,
        )
        self.video_discr = MoCoGANVideoDiscriminator(
            n_channels=img_channels,
            n_output_neurons=1,
            bn_use_gamma=True,
            use_noise=True,
            noise_sigma=0.1,
            image_size=img_resolution,
            num_t_paddings=cfg.video_discr_num_t_paddings,
        )
        self.video_discr.apply(weights_init)

    def params_with_lr(self, lr: float) -> List[Dict]:
        return [
            {'params': self.image_discr.parameters()},
            {'params': self.video_discr.parameters(), 'lr': self.cfg.video_discr_lr_multiplier * lr}
        ]

    def forward(self, img: Tensor, c: Tensor, t: Tensor, **img_discr_kwargs) -> Tuple[Tensor, "None"]:
        """
        - img has shape [batch_size * num_frames_per_video, c, h, w]
        - c has shape [batch_size, c_dim]
        - t has shape [batch_size, num_frames_per_video]
        """
        batch_size, num_frames_per_video = t.shape
        image_logits = self.image_discr(img, c, t, **img_discr_kwargs)['image_logits'] # [batch_size * num_frames]

        # Preparing input for the video discriminator
        videos = img.view(batch_size, num_frames_per_video, *img.shape[1:]) # [batch_size, t, c, h, w]
        videos = videos.permute(0, 2, 1, 3, 4).contiguous() # [batch_size, c, t, h, w]
        video_logits = self.video_discr(videos) # (num_subdiscrs, num_layers, [batch_size, 1, out_t, out_h, out_w])

        # We return a tuple for backward compatibility
        return {'image_logits': image_logits, 'video_logits': video_logits.flatten(start_dim=1)}

#----------------------------------------------------------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d,
                                       affine=False,
                                       track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

#----------------------------------------------------------------------------

@persistence.persistent_class
class VideoDiscriminator(nn.Module):
    def __init__(self,
                 num_input_channels,
                 ndf=64,
                 n_layers=3,
                 n_frames_per_sample=16,
                 norm_layer=nn.InstanceNorm3d,
                 num_sub_discrs=2,
                 get_intermediate_feat=True):

        super().__init__()
        self.num_sub_discrs = num_sub_discrs
        self.n_layers = n_layers
        self.get_intermediate_feat = get_intermediate_feat
        ndf_max = 64

        for i in range(num_sub_discrs):
            block = SubVideoDiscriminator(
                num_input_channels=num_input_channels,
                ndf=min(ndf_max, ndf * (2 ** (num_sub_discrs - 1 - i))),
                n_layers=n_layers,
                norm_layer=norm_layer,
                get_intermediate_feat=get_intermediate_feat)

            if get_intermediate_feat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(block, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), block.model)

        stride = 2 if n_frames_per_sample > 16 else [1, 2, 2]
        self.downsample = nn.AvgPool3d(
            3,
            stride=stride,
            padding=[1, 1, 1],
            count_include_pad=False
        )

    def singleD_forward(self, model, input):
        if self.get_intermediate_feat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, x):
        result = []
        x = x

        for block_idx in range(self.num_sub_discrs):
            if self.get_intermediate_feat:
                model = [getattr(self, 'scale' + str(self.num_sub_discrs - 1 - block_idx) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(self.num_sub_discrs - 1 - block_idx))
            result.append(self.singleD_forward(model, x))

            if block_idx != (self.num_sub_discrs - 1):
                x = self.downsample(x)

        return result

#----------------------------------------------------------------------------

@persistence.persistent_class
class SubVideoDiscriminator(nn.Module):
    def __init__(self,
                 num_input_channels,
                 ndf=64,
                 n_layers=3,
                 norm_layer=nn.InstanceNorm3d,
                 get_intermediate_feat=True):

        super().__init__()
        self.get_intermediate_feat = get_intermediate_feat
        self.n_layers = n_layers

        kernel_size = 4
        padw = int(np.ceil((kernel_size - 1.0) / 2))

        sequence = [[
            nn.Conv3d(num_input_channels, ndf, kernel_size=kernel_size, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kernel_size, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kernel_size, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[
            nn.Conv3d(nf, 1, kernel_size=kernel_size, stride=1, padding=padw)
        ]]

        if get_intermediate_feat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            self.model = nn.Sequential(*[s for ss in sequence for s in ss])

    def forward(self, x):
        if self.get_intermediate_feat:
            res = [x]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(x)

#----------------------------------------------------------------------------

class MoCoGANVideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64, image_size: int=64, num_t_paddings: int=0):
        super(MoCoGANVideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        layers = [
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(2 if num_t_paddings > 0 else 0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(2 if num_t_paddings > 1 else 0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(2 if num_t_paddings > 2 else 0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 4, ndf * 8, 4, stride=(1, 2, 2), padding=(2 if num_t_paddings > 3 else 0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if image_size == 256:
            layers.extend([
                Noise(use_noise, sigma=noise_sigma),
                nn.Conv3d(ndf * 8, ndf * 8, 3, stride=(1, 1, 1), padding=(1 + (1 if num_t_paddings > 4 else 0), 1, 1), bias=False),
                nn.BatchNorm3d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                Noise(use_noise, sigma=noise_sigma),
                nn.Conv3d(ndf * 8, ndf * 8, 3, stride=(1, 1, 1), padding=(1 + (1 if num_t_paddings > 5 else 0), 1, 1), bias=False),
                nn.BatchNorm3d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        layers.extend([
            nn.Conv3d(ndf * 8, n_output_neurons, kernel_size=4, stride=1, padding=(2 if num_t_paddings > 5 else 0, 0, 0), bias=False),
        ])

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input).squeeze()

#----------------------------------------------------------------------------

class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()

        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * torch.randn_like(x)
        return x

#----------------------------------------------------------------------------
