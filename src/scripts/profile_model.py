"""
This script computes imgs/sec for a generator in the eval mode
for different batch sizes
"""
import sys; sys.path.extend(['..', '.', 'src'])
import time

import numpy as np
import torch
import torch.nn as nn
import hydra
from hydra.experimental import initialize
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch.autograd.profiler as profiler

from src import dnnlib
from src.infra.utils import recursive_instantiate


DEVICE = 'cuda'
BATCH_SIZES = [32]
NUM_WARMUP_ITERS = 5
NUM_PROFILE_ITERS = 25


def instantiate_G(cfg: DictConfig) -> nn.Module:
    G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    G_kwargs.synthesis_kwargs.channel_base = int(cfg.model.generator.get('fmaps', 0.5) * 32768)
    G_kwargs.synthesis_kwargs.channel_max = 512
    G_kwargs.mapping_kwargs.num_layers = cfg.model.generator.get('mapping_net_n_layers', 2)
    if cfg.get('num_fp16_res', 0) > 0:
        G_kwargs.synthesis_kwargs.num_fp16_res = cfg.num_fp16_res
        G_kwargs.synthesis_kwargs.conv_clamp = 256
    G_kwargs.cfg = cfg.model.generator
    G_kwargs.c_dim = 0
    G_kwargs.img_resolution = cfg.get('resolution', 256)
    G_kwargs.img_channels = 3

    G = dnnlib.util.construct_class_by_name(**G_kwargs).eval().requires_grad_(False).to(DEVICE)

    return G


@torch.no_grad()
def profile_for_batch_size(G: nn.Module, cfg: DictConfig, batch_size: int):
    z = torch.randn(batch_size, G.z_dim, device=DEVICE)
    c = torch.zeros(batch_size, G.c_dim, device=DEVICE)
    t = torch.zeros(batch_size, 2, device=DEVICE)
    times = []

    for i in tqdm(range(NUM_WARMUP_ITERS), desc='Warming up'):
        torch.cuda.synchronize()
        fake_img = G(z, c=c, t=t).contiguous()
        y = fake_img[0, 0, 0, 0].item() # sync
        torch.cuda.synchronize()

    time.sleep(1)

    torch.cuda.reset_peak_memory_stats()

    with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        for i in tqdm(range(NUM_PROFILE_ITERS), desc='Profiling'):
            torch.cuda.synchronize()
            start_time = time.time()
            with profiler.record_function("forward"):
                fake_img = G(z, c=c, t=t).contiguous()
                y = fake_img[0, 0, 0, 0].item() # sync
            torch.cuda.synchronize()
            times.append(time.time() - start_time)

    torch.cuda.empty_cache()
    num_imgs_processed = len(times) * batch_size
    total_time_spent = np.sum(times)
    bandwidth = num_imgs_processed / total_time_spent
    summary = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)

    print(f'[Batch size: {batch_size}] Mean: {np.mean(times):.05f}s/it. Std: {np.std(times):.05f}s')
    print(f'[Batch size: {batch_size}] Imgs/sec: {bandwidth:.03f}')
    print(f'[Batch size: {batch_size}] Max mem: {torch.cuda.max_memory_allocated(DEVICE) / 2**30:<6.2f} gb')

    return bandwidth, summary


@hydra.main(config_path="../../configs", config_name="config.yaml")
def profile(cfg: DictConfig):
    recursive_instantiate(cfg)
    G = instantiate_G(cfg)
    bandwidths = []
    summaries = []
    print(f'Number of parameters: {sum(p.numel() for p in G.parameters())}')

    for batch_size in BATCH_SIZES:
        bandwidth, summary = profile_for_batch_size(G, cfg, batch_size)
        bandwidths.append(bandwidth)
        summaries.append(summary)

    best_batch_size_idx = int(np.argmax(bandwidths))
    print(f'------------ Best batch size is {BATCH_SIZES[best_batch_size_idx]} ------------')
    print(summaries[best_batch_size_idx])


if __name__ == '__main__':
    profile()
