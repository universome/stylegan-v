"""
Given a dataset of images, it (optionally crops it) and embeds into the model
Also optionally generates random videos from the found w
"""

import sys; sys.path.extend(['.', 'src'])
import os
import re
import json
import random
from typing import List, Optional, Callable
from typing import List

from PIL import Image
import click
from src import dnnlib
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from torch import Tensor
import torchvision.transforms.functional as TVF
from torchvision.utils import save_image

import legacy
from src.training.logging import generate_videos, save_video_frames_as_mp4, save_video_frames_as_frames
from src.torch_utils import misc

#----------------------------------------------------------------------------

def project(
    _sentinel=None,
    G: Callable=None,
    vgg16: nn.Module=None,
    target_images: List[Tensor]=None,
    device: str='cuda',
    use_w_init: bool=False,
    use_motion_init: bool=False,
    w_avg_samples = 10000,
    num_steps = 1000,
    initial_learning_rate = 0.1,
    initial_noise_factor = 0.05,
    noise_ramp_length = 0.75,
    lr_rampdown_length = 0.25,
    lr_rampup_length = 0.05,
    #regularize_noise_weight = 1e5,
    regularize_noise_weight = 0.0001,
    motion_reg_type: str=None,
):
    num_videos = len(target_images)

    # misc.assert_shape(target_images, [None, G.img_channels, G.img_resolution, G.img_resolution])
    G = G.eval().requires_grad_(False).to(device) # type: ignore

    c = torch.zeros(num_videos, G.c_dim, device=device)
    ts = torch.zeros(num_videos, 1, device=device)

    # Compute w stats.
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # img_mean = G.synthesis(
    #     ws=torch.from_numpy(w_avg).repeat(1, G.num_ws, 1).to(device),
    #     c=c[0], t=ts[[0]],
    # )
    # img_mean = (img_mean * 0.5 + 0.5).cpu().detach()
    # TVF.to_pil_image(img_mean[0]).save('/tmp/data/mean.png')
    # print('saved!')

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_features = []
    for img in target_images:
        img = img.to(device).to(torch.float32).unsqueeze(0) * 255.0
        if img.shape[2] > 256:
            img = F.interpolate(img, size=(256, 256), mode='area')
        target_features.append(vgg16(img, resize_images=False, return_lpips=True).squeeze(0))
    target_features = torch.stack(target_features) # [num_images, lpips_dim]

    if use_w_init:
        w_opt = find_w_init() # [num_videos, 1, w_dim]
        w_opt = w_opt.detach().requires_grad_(True) # [num_videos, num_ws, w_dim]
    else:
        w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
        w_opt = w_opt.repeat(num_videos, G.num_ws, 1).detach().requires_grad_(True) # [num_videos, num_ws, w_dim]

    # w_opt_to_ws = lambda w_opt: torch.cat([w_opt[:, [0]].repeat(1, G.num_ws // 2, 1), w_opt[:, 1:]], dim=1)

    # Trying a lot of motions to find which one works best
    if use_motion_init:
        motion_z_opt = select_motions(motion_codes)
    else:
        motion_z_opt = G.synthesis.motion_encoder(c=c, t=ts)['motion_z']
        # motion_z_opt.data = torch.randn_like(motion_z_opt.data) * 1e-3

    motion_z_opt.requires_grad_(True)

    w_result = torch.zeros([num_steps] + list(w_opt.shape), dtype=torch.float32, device=device)
    # optimizer = torch.optim.Adam([w_opt] + [motion_z_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
    optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

    for step in tqdm(range(num_steps)):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        #ws = w_opt_to_ws(w_opt + w_noise)
        #ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        #synth_images = G.synthesis(ws, c=c, t=ts, motion_z=motion_z_opt + torch.randn_like(motion_z_opt) * w_noise_scale)
        synth_images = G.synthesis(ws, c=c, t=ts, motion_z=motion_z_opt)
        #synth_images = G.synthesis(ws, c=c, t=ts)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images * 0.5 + 0.5) * 255.0
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        if motion_reg_type is None:
            reg_loss = 0.0
        elif motion_reg_type == "norm":
            reg_loss = motion_z_opt.norm(dim=2).mean()
        elif motion_reg_type == "dist":
            reg_loss = motion_z_opt.mean().pow(2) + (motion_z_opt.var() - 1).pow(2)
        elif motion_reg_type == "sg2":
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
        else:
            raise NotImplementedError(f"Uknown motion_reg_type: {motion_reg_type}")

        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Save projected W for each optimization step.
        w_result[step] = w_opt.detach()

        # Normalize noise.
    #     with torch.no_grad():
    #         for buf in motion_z_opt.values():
    #             buf -= buf.mean()
    #             buf *= buf.square().mean().rsqrt()

    return w_result, motion_z_opt

#----------------------------------------------------------------------------

@torch.no_grad()
def find_motions_init(G: Callable, vgg16: nn.Module, target_features: Tensor, c: Tensor, t: Tensor, num_motions_to_try: int=128):
    motions = G.synthesis.motion_encoder(
        c=c.repeat_interleave(num_motions_to_try, dim=0),
        t=t.repeat_interleave(num_motions_to_try, dim=0))['motion_z'] # [num_videos * num_motions_to_try, ...]

    synth_images = G.synthesis(
        w_opt.repeat_interleave(num_motions_to_try, dim=0),
        c=c.repeat_interleave(num_motions_to_try, dim=0),
        t=t.repeat_interleave(num_motions_to_try, dim=0),
        motion_z=motions)

    if synth_images.shape[2] > 256:
        synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

    synth_images = (synth_images * 0.5 + 0.5) * 255.0
    synth_features = vgg16(synth_images, resize_images=False, return_lpips=True) # [num_videos * num_motions_to_try, ...]
    dist = (target_features.repeat_interleave(num_motions_to_try, dim=0) - synth_features).square().sum(dim=1) # [num_videos * num_motions_to_try]
    best_motions_idx = dist.view(num_videos, num_motions_to_try).argmin(dim=1) # [num_videos]
    motion_z_opt = motions[best_motions_idx] # [num_videos, ...]

    return motion_z_opt

#----------------------------------------------------------------------------

@torch.no_grad()
def find_w_init(G: Callable, vgg16: nn.Module, target_features: Tensor, c: Tensor, t: Tensor, l: Tensor, num_w_to_try: int=128):
    z = torch.randn(num_videos * num_w_to_try, G.z_dim, device=device)
    w = G.mapping(z=z, c=None)  # [N, L, C]

    synth_images = G.synthesis(
        ws=w,
        c=c.repeat_interleave(num_w_to_try, dim=0),
        t=t.repeat_interleave(num_w_to_try, dim=0))
    if synth_images.shape[2] > 256:
        synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
    synth_images = (synth_images * 0.5 + 0.5) * 255.0
    synth_features = vgg16(synth_images, resize_images=False, return_lpips=True) # [num_videos * num_motions_to_try, ...]
    dist = (target_features.repeat_interleave(num_w_to_try, dim=0) - synth_features).square().sum(dim=1) # [num_videos * num_motions_to_try]
    best_w_idx = dist.view(num_videos, num_w_to_try).argmin(dim=1) # [num_videos]
    w_opt = w[best_w_idx] # [num_videos, num_ws, w_dim]

    return w_opt

#----------------------------------------------------------------------------

@torch.no_grad()
def load_target_images(img_paths: List[os.PathLike], extract_faces: bool=False, ref_image: Tensor=None):
    images = [Image.open(f) for f in tqdm(img_paths, desc='Loading images')]

    if extract_faces:
        images = extract_faces_from_images(imgs=images, ref_image=ref_image)
        for p, img in zip(img_paths, images):
            img.save('/tmp/data/faces_extracted/' + os.path.basename(p), q=95)
        assert False
        # grid = torch.stack([TVF.to_tensor(x) for x in images])
        # grid = utils.make_grid(grid, nrow=8)
        # save_image(grid, f'/tmp/data/faces_extracted.png')
        # print('Saved the extracted images!')

    # images = [x[:, 200:-400, 450:-200] for x in images]
    images = [TVF.to_tensor(x) for x in images]
    images = [TVF.resize(x, size=(256, 256)) for x in images]

    return images

#----------------------------------------------------------------------------

@torch.no_grad()
def extract_faces_from_images(_sentinel=None, imgs: List=None, ref_image: "Image"=None, device: str='cuda'):
    assert _sentinel is None
    try:
        import face_alignment
    except ImportError:
        raise ImportError("To project images with alignment, you need to install the `face_alignment` library.")

    SELECTED_LANDMARKS = [38, 44]
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

    ref_landmarks = fa.get_landmarks_from_image(np.array(ref_image))[0][SELECTED_LANDMARKS] # [2, 2]
    landmarks = [fa.get_landmarks_from_image(np.array(x))[0][SELECTED_LANDMARKS] for x in imgs] # [num_imgs, 2, 2]
    ref_dist = ((ref_landmarks[0] - ref_landmarks[1]) ** 2).sum() ** 0.5 # [1]
    dists = [((p[0] - p[1]) ** 2).sum() ** 0.5 for p in landmarks] # [num_imgs]
    resize_ratios = [ref_dist / d for d in dists] # [num_imgs]
    new_sizes = [(int(r * x.size[1]), int(r * x.size[0])) for r, x in zip(resize_ratios, imgs)]
    imgs_resized = [TVF.resize(x, size=s, interpolation=Image.LANCZOS) for x, s in zip(imgs, new_sizes)] # [num_imgs, Image]
    bbox_left = [p[0][0] * r - ref_landmarks[0][0] for p, r in zip(landmarks, resize_ratios)]
    bbox_top = [p[0][1] * r - ref_landmarks[0][1] for p, r in zip(landmarks, resize_ratios)]

    out = [x.crop(box=(l, t, l + ref_image.size[0], t + ref_image.size[1])) for x, l, t in zip(imgs_resized, bbox_left, bbox_top)]

    return out

#----------------------------------------------------------------------------

def pad_box_to_square(left, upper, right, lower):
    h = lower - upper
    w = right - left

    if h == w:
        return left, upper, right, lower
    elif w > h:
        diff = w - h
        assert False, "Not implemented"
    else:
        pad = (h - w) // 2

        return (left - pad, upper, right + pad, lower)

#----------------------------------------------------------------------------

def add_margins(box, margin, width: int=float('inf'), height: int=float('inf')):
    left, upper, right, lower = box

    return (
        max(0, left - margin[0]),
        max(0, upper - margin[1]),
        min(width, right + margin[2]),
        min(height, lower + margin[3]),
    )

#----------------------------------------------------------------------------

def add_top_margin(box, margin_ratio: float=0.0):
    left, upper, right, lower = box
    height = lower - upper
    margin = int(height * margin_ratio)

    return (left, max(0, upper - margin), right, lower)

#----------------------------------------------------------------------------

def save_edited_w(
        _sentinel=None,
        G: Callable=None,
        w_outdir: os.PathLike=None,
        samples_outdir: os.PathLike=None,
        img_names: List[str]=None,
        stack_samples: bool=False,
        num_frames: int = 16,
        each_nth_frame: int = 3,
        all_w: Tensor=None,
        all_motion_z: Tensor=None,
        stacked_samples_out_path: os.PathLike=None,
    ):
    assert _sentinel is None

    # w_outdir = os.path.join(os.path.basename(images_dir))

    os.makedirs(w_outdir, exist_ok=True)
    num_videos = len(img_names)
    device = all_w.device

    if not stack_samples:
        os.makedirs(samples_outdir, exist_ok=True)
    else:
        all_samples = []

    # Generate samples from the given w and save them.
    with torch.no_grad():
        z = torch.randn(num_videos, G.z_dim, device=device) # [num_videos, z_dim]
        c = torch.zeros(num_videos, G.c_dim, device=device) # [num_videos, c_dim]

        for i, w in enumerate(all_w):
            torch.save(w.cpu(), os.path.join(w_outdir, f'{img_names[i]}_w.pt'))

            if all_motion_z is None:
                motion_z = None
            else:
                motion_z = all_motion_z[i] # [...<any>...]
                torch.save(motion_z.cpu(), os.path.join(w_outdir, f'{img_names[i]}_motion.pt'))
                motion_z = motion_z.unsqueeze(0).to(device) # [1, ...<any>...]
                motion_z = torch.randn_like(motion_z)

            w = w.unsqueeze(0).to(device) # [1, num_ws, w_dim]
            t = torch.linspace(0, num_frames * (1 + each_nth_frame), num_frames, device=device).unsqueeze(0)
            imgs = G.synthesis(w, c=c[[i]]], t=t, motion_z=motion_z)
            imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
            grid = utils.make_grid(imgs, nrow=num_frames).cpu()

            if stack_samples:
                all_samples.append(grid)
            else:
                # TVF.to_pil_image(grid).save(os.path.join(samples_outdir, img_names[i]) + '.jpg', q=95)
                save_image(grid, os.path.join(samples_outdir, img_names[i]) + '.png')

    if stack_samples:
        main_grid = torch.stack(all_samples) # [num_videos, c, h, w * num_frames]
        main_grid = utils.make_grid(main_grid, nrow=1)
        # TVF.to_pil_image(main_grid).save(f'{images_dir}.jpg', q=95)
        save_image(main_grid, stacked_samples_out_path)

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network_pkl', help='Network pickle filename', metavar='PATH')
@click.option('--networks_dir', help='Network pickles directory', metavar='PATH')
# @click.option('--truncation_psi', type=float, help='Truncation psi', default=1.0, show_default=True)
# @click.option('--noise_mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
# @click.option('--same_motion_codes', type=bool, help='Should we use the same motion codes for all videos?', default=False, show_default=True)
@click.option('--seed', type=int, help='Random seed', default=42, metavar='DIR')
@click.option('--images_dir', help='Where to save the output images', type=str, required=True, metavar='DIR')
# @click.option('--save_as_mp4', help='Should we save as independent frames or mp4?', type=bool, default=False, metavar='BOOL')
# @click.option('--video_len', help='Number of frames to generate', type=int, default=16, metavar='INT')
# @click.option('--fps', help='FPS for mp4 saving', type=int, default=25, metavar='INT')
# @click.option('--as_grids', help='Save videos as grids', type=bool, default=False, metavar='BOOl')
@click.option('--zero_periods', help='Zero-out periods predictor?', default=False, type=bool, metavar='BOOL')
@click.option('--num_weights_to_slice', help='Number of high-frequency coords to remove.', default=0, type=int, metavar='INT')
@click.option('--use_w_init', help='Init w by LPIPS.', default=False, type=bool, metavar='BOOL')
@click.option('--use_motion_init', help='Init motions by LPIPS.', default=False, type=bool, metavar='BOOL')
@click.option('--motion_reg_type', help='Type of the regularization for motion', default=None, type=str, metavar='STR')
@click.option('--num_steps', help='Number of the optimization steps to perform.', default=1000, type=int, metavar='INT')
@click.option('--stack_samples', help='When saving, should we stack samples together?', default=False, type=bool, metavar='BOOL')
@click.option('--extract_faces', help='Use FaceNet to extract the face?', default=False, type=bool, metavar='BOOL')

def main(
    ctx: click.Context,
    network_pkl: str,
    networks_dir: str,
    seed: int,
    images_dir: str,
    # save_as_mp4: bool,
    # video_len: int,
    # fps: int,
    # as_grids: bool,
    zero_periods: bool,
    num_weights_to_slice: int,
    use_w_init: bool,
    use_motion_init: bool,
    motion_reg_type: str,
    num_steps: int,
    stack_samples: bool,
    extract_faces: bool,
):
    if network_pkl is None:
        output_regex = "^network-snapshot-\d{6}.pkl$"
        ckpt_regex = re.compile("^network-snapshot-\d{6}.pkl$")
        # ckpts = sorted([f for f in os.listdir(networks_dir) if ckpt_regex.match(f)])
        # network_pkl = os.path.join(networks_dir, ckpts[-1])
        metrics_file = os.path.join(networks_dir, 'metric-fvd2048_16f.jsonl')
        with open(metrics_file, 'r') as f:
            snapshot_metrics_vals = [json.loads(line) for line in f.read().splitlines()]
        best_snapshot = sorted(snapshot_metrics_vals, key=lambda m: m['results']['fvd2048_16f'])[0]
        network_pkl = os.path.join(networks_dir, best_snapshot['snapshot_pkl'])
        print(f'Using checkpoint: {network_pkl} with FVD16 of', best_snapshot['results']['fvd2048_16f'])
        # Selecting a checkpoint with the best score
    else:
        assert networks_dir is None, "Cant have both parameters: network_pkl and networks_dir"

    print('Loading networks from "%s"...' % network_pkl, end='')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
    print('Loaded!')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if zero_periods:
        G.synthesis.motion_encoder.time_encoder.periods_predictor.weight.data.zero_()

    if num_weights_to_slice > 0:
        G.synthesis.motion_encoder.time_encoder.weights[:, -num_weights_to_slice:] = 0.0

    img_paths = sorted([os.path.join(images_dir, p) for p in os.listdir(images_dir) if p.endswith('.jpg')])
    img_names = [n[:n.rfind('.')] for n in [os.path.basename(p) for p in img_paths]]
    target_images = load_target_images(img_paths, extract_faces, ref_image=Image.open('/tmp/data/mean.png')) # [b, c, h, w]

    assert G.c_dim == 0, "G.c_dim > 0 is not supported"

    w_all_iters, motion_z_final = project(
        G=G,
        target_images=target_images,
        num_steps=num_steps,
        device=device,
        use_w_init=use_w_init,
        use_motion_init=use_motion_init,
        motion_reg_type=motion_reg_type,
    ) # [num_videos, num_ws, w_dim]

    save_edited_w(
        G=G,
        w_outdir = f'{images_dir}_projected',
        samples_outdir = f'{images_dir}_projected_samples',
        img_names=img_names,
        stack_samples=stack_samples,
        all_w = w_all_iters[-1],
        all_motion_z = motion_z_final,
        stacked_samples_out_path = f'{images_dir}.png'
    )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
