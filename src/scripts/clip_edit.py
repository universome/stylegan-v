# import sys; sys.path.extend(['.', 'src', '/home/skoroki/StyleCLIP'])
import argparse
import math
import os
from typing import List
import json
import re
import random
import yaml
import itertools

import torchvision
from torch import optim
from PIL import Image
import click
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
from torch import Tensor

from src.deps.facial_recognition.model_irse import Backbone

try:
    import clip
except ImportError:
    raise ImportError(
        "To edit videos with CLIP, you need to install the `clip` library. " \
        "Please follow the instructions in https://github.com/openai/CLIP")

from src import dnnlib
import legacy
from src.scripts.project import save_edited_w


#----------------------------------------------------------------------------

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

#----------------------------------------------------------------------------

class CLIPLoss(torch.nn.Module):
    """
    Copy-pasted and adapted from StyleCLIP
    """
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        #self.upsample = torch.nn.Upsample(scale_factor=7)
        #self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

    def forward(self, image, text):
        #image = self.avg_pool(self.upsample(image))
        #print('shape', image.shape, text.shape)
        image = F.interpolate(image, size=(224, 224), mode='area')
        similarity = 1 - self.model(image, text)[0] / 100
        similarity = similarity.diag()

        return similarity

#----------------------------------------------------------------------------

class IDLoss(nn.Module):
    """
    Copy-pasted from StyleCLIP
    """
    def __init__(self):
        super(IDLoss, self).__init__()
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        with dnnlib.util.open_url(Backbone.WEIGHTS_URL, verbose=True) as f:
            ir_se50_weights = torch.load(f)
        self.facenet.load_state_dict(ir_se50_weights)
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0

        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target

        return loss / n_samples

#----------------------------------------------------------------------------

def run_edit_optimization(
    _sentinel=None,
    G: nn.Module=None,
    w_orig: Tensor=None,
    descriptions: List[str]=None,
    # ckpt: float="stylegan2-ffhq-config-f.pt",
    lr: float=0.1,
    num_steps: int=40,
    l2_lambda: float=0.001,
    id_lambda: float=0.005,
    # latent_path: float=latent_path,
    # truncation: float=0.7,
    # save_intermediate_image_every: float=1 if create_video else 20,
    # results_dir: float="results",
    mask: float=None,
    mask_lambda: float=0.0,
    verbose: bool=False,
) -> Tensor:
    assert _sentinel is None
    # text_inputs = torch.cat([clip.tokenize(d) for d in descriptions]).to(device)
    num_prompts = len(descriptions)
    num_images = len(w_orig)
    device = w_orig.device

    text_inputs = clip.tokenize(descriptions).to(device) # [num_prompts, 77]
    text_inputs = text_inputs.repeat_interleave(len(w_orig), dim=0) # [num_prompts * num_images, 77]

    c = torch.zeros(num_prompts * num_images, 0, device=device)
    ts = torch.zeros(num_prompts * num_images, 1, device=device)
    w_orig = w_orig.repeat(num_prompts, 1, 1) # [num_prompts * num_images, num_ws, w_dim]

    with torch.no_grad():
        img_orig = G.synthesis(ws=w_orig, c=c, t=ts) # [num_prompts * num_images, 3, c, h, w]

    w = w_orig.detach().clone() # [num_prompts * num_images, num_ws, w_dim]
    w.requires_grad = True

    if mask_lambda > 0:
        target_image = img_orig * (1 - mask) # [num_prompts * num_images, 3, c, h, w]
        #target_image = img_orig[:, :, -128:, :128]
        target_image = (target_image * 0.5 + 0.5) * 255.0 # [num_prompts * num_images, 3, c, h, w]
        if target_image.shape[2] > 256:
            target_image = F.interpolate(target_image, size=(256, 256), mode='area')
        target_features = vgg16(target_image, resize_images=False, return_lpips=True)
        #dist = (target_features - synth_features).square().sum()
    else:
        target_features = None

    clip_loss = CLIPLoss()
    id_loss = IDLoss()
    optimizer = optim.Adam([w], lr=lr)

    if verbose:
        pbar = tqdm(range(num_steps))
    else:
        pbar = range(num_steps)

    for curr_iter in pbar:
        curr_lr = get_lr(curr_iter / num_steps, lr)
        # optimizer.param_groups[0]["lr"] = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr

        #img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=work_in_stylespace)
        img_gen = G.synthesis(ws=w, c=c, t=ts) # [num_prompts * num_images, 3, c, h, w]

        if mask_lambda > 0:
            raise NotImplementedError
            synth_image = img_gen * (1 - mask)
            #synth_image = img_gen[:, :, -128:, :128]
            synth_image = (synth_image * 0.5 + 0.5) * 255.0
            if synth_image.shape[2] > 256:
                synth_image = F.interpolate(synth_image, size=(256, 256), mode='area')
            synth_features = vgg16(synth_image, resize_images=False, return_lpips=True)
            mask_loss = (target_features - synth_features).square().sum()
        else:
            mask_loss = 0

        if not mask is None:
            img_gen = img_gen * mask.unsqueeze(0) # [num_prompts * num_images, 3, c, h, w]

        c_loss = clip_loss(img_gen, text_inputs) # [num_prompts * num_images]

        if id_lambda > 0:
            i_loss = id_loss(img_gen, img_orig)
        else:
            i_loss = 0

        l2_loss = ((w_orig - w) ** 2) # [1]
        loss = c_loss.sum() + l2_lambda * l2_loss.sum() + id_lambda * i_loss + mask_lambda * mask_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            pbar.set_description((f"loss: {loss.item():.4f};"))

    final_result = torch.stack([img_orig, img_gen]) # [2, num_prompts * num_images, c, h, w]

    return final_result, w

    # x, new_w = main(args)

    # pair = torch.cat([img for img in x], dim=2)
    # TVF.to_pil_image((pair.cpu().detach() * 0.5 + 0.5).clamp(0, 1))

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network_pkl', help='Network pickle filename', metavar='PATH')
@click.option('--networks_dir', help='Network pickles directory', metavar='PATH')
# @click.option('--truncation_psi', type=float, help='Truncation psi', default=1.0, show_default=True)
# @click.option('--noise_mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
# @click.option('--same_motion_codes', type=bool, help='Should we use the same motion codes for all videos?', default=False, show_default=True)
@click.option('--w_dir', help='A directory leading to latent codes.', type=str, required=False, metavar='DIR')
@click.option('--results_dir', help='A directory to save the results in.', type=str, required=False, metavar='DIR')
@click.option('--truncation_psi', help='If we use new w, what truncation to use.', type=float, required=False, metavar='FLOAT', default=1.0)
@click.option('--num_w', help='If we use new w, how many to sample?', type=int, required=False, metavar='FLOAT', default=16)
@click.option('--prompts', help='A path to prompts or a string of prompts.', type=str, required=True, metavar='DIR')
@click.option('--seed', type=int, help='Random seed', default=42, metavar='DIR')
@click.option('--zero_periods', help='Zero-out periods predictor?', default=False, type=bool, metavar='BOOL')
@click.option('--num_weights_to_slice', help='Number of high-frequency coords to remove.', default=0, type=int, metavar='INT')
@click.option('--num_steps', help='Number of the optimization steps to perform.', default=40, type=int, metavar='INT')
@click.option('--stack_samples', help='When saving, should we stack samples together?', default=False, type=bool, metavar='BOOL')
# l2_lambda=0.001,
# id_lambda=0.005,
# l2_lambda=0.0005,
# id_lambda=0.0,
@click.option('--l2_lambda', help='L2 loss coef', default=0.001, type=float, metavar='FLOAT')
@click.option('--id_lambda', help='ID loss coef', default=0.005, type=float, metavar='FLOAT')
@click.option('--lr', help='Learning rate', default=0.1, type=float, metavar='FLOAT')
@click.option('--mask_lambda', help='If we use a mask, specify the loss coef', default=0.0, type=float, metavar='FLOAT')
@click.option('--use_id_lambda', help='Should we use id lambda in HPO?', default=False, type=bool, metavar='BOOL')

def main(
    ctx: click.Context,
    network_pkl: str,
    networks_dir: str,
    seed: int,
    w_dir: str,
    results_dir: str,
    truncation_psi: float,
    num_w: int,
    # save_as_mp4: bool,
    # video_len: int,
    # fps: int,
    # as_grids: bool,
    zero_periods: bool,
    num_weights_to_slice: int,
    num_steps: int,
    stack_samples: bool,
    l2_lambda: float,
    id_lambda: float,
    lr: float,
    prompts: str,
    mask_lambda: float,
    use_id_lambda: bool,
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

    # description = "Bright sunny sky and mountains far away"
    # experiment_type = 'edit' #@param ['edit', 'free_generation']
    # mask = torch.zeros(3, 256, 256, device=device)
    # mask[:, :, 64+32 : 128+32] = 1.0
    # mask[:, :-128, :] = 1.0
    # mask[:, :, 128:] = 1.0

    if w_dir is None:
        print('Sampling new w')
        z = torch.randn(num_w, G.z_dim, device=device)
        c = torch.zeros(len(z), G.c_dim, device=device)
        w_orig = G.mapping(z=z, c=c, truncation_psi=truncation_psi)
        os.makedirs(results_dir, exist_ok=True)
        torch.save(w_orig.cpu(), f'{results_dir}_w_orig.pt')
        w_save_dir = os.path.join(results_dir, 'w_edit')
        samples_save_dir = os.path.join(results_dir, 'edited_samples')
    else:
        w_paths = sorted([os.path.join(w_dir, f) for f in os.listdir(w_dir) if f.endswith('_w.pt')])
        w_names = [os.path.basename(f) for f in w_paths]
        w_orig = [torch.load(f) for f in w_paths]
        w_orig = torch.stack(w_orig).to(device) # [num_images, num_ws, w_dim]
        w_save_dir = f'{w_dir}_edited_w'
        samples_save_dir = f'{w_dir}_edited_samples'

    os.makedirs(w_save_dir, exist_ok=True)
    os.makedirs(samples_save_dir, exist_ok=True)

    print(f'Loading prompts from file: {prompts}')
    with open(prompts, 'r') as f:
        descs_dict = yaml.load(f)
        edit_names, descriptions = list(zip(*descs_dict.items()))
        edit_names = edit_names
        descriptions = descriptions

    del id_lambda, num_steps, l2_lambda
    l2_lambdas = [1000000.0, 0.0025, 0.001, 0.00025, 0.0005, 0.0001]
    if use_id_lambda:
        id_lambdas = [0.005, 0.0025, 0.001, 0.00025, 0.0005, 0.0001, 0.0]
    else:
        id_lambdas = [0.0]
    all_num_steps = [40]

    for curr_edit_name, curr_prompt in zip(edit_names, descriptions):
        all_images = []
        all_w_edited = []

        for l2_lambda, id_lambda, num_steps in tqdm(list(itertools.product(l2_lambdas, id_lambdas, all_num_steps)), desc=f'Performing HPO for {curr_edit_name}'):
            final_image, w_edited = run_edit_optimization(
                G=G,
                w_orig=w_orig,
                descriptions=[curr_prompt],
                # ckpt="stylegan2-ffhq-config-f.pt",
                lr=lr,
                num_steps=num_steps,
                l2_lambda=l2_lambda,
                id_lambda=id_lambda,
                mask_lambda=mask_lambda,
                verbose=False,
                # latent_path=latent_path,
                # truncation=0.7,
                # mask=None,
                # mask_lambda=0.1,
            )

            all_images.extend((final_image[1].cpu() * 0.5 + 0.5).clamp(0, 1))
            all_w_edited.append({
                "w_edit": w_edited.cpu(),
                "l2_lambda": l2_lambda,
                "id_lambda": id_lambda,
                "num_steps": num_steps,
                "prompt": curr_prompt,
                "edit_name": curr_edit_name,
            })

            # img_names = [f'{w_name}_{edit_name}' for edit_name in edit_names for w_name in w_names]

            # save_edited_w(
            #     G=G,
            #     w_outdir = f'{w_dir}_edited',
            #     samples_outdir = f'{w_dir}_projected_samples',
            #     img_names=img_names,
            #     stack_samples=stack_samples,
            #     all_w = w_edited,
            #     all_motion_z = None,
            #     stacked_samples_out_path = f'{w_dir}_edited_samples.png'
            # )

        torch.save(all_w_edited, f"{w_save_dir}/{curr_edit_name}_w.pt")
        grid = utils.make_grid(torch.stack(all_images), nrow=len(w_orig))
        print('savig intp', f"{samples_save_dir}/{curr_edit_name}.png")
        save_image(grid, f"{samples_save_dir}/{curr_edit_name}.png")

    print('Done!')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
