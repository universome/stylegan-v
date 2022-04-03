# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""

import sys; sys.path.extend(['.'])
import os
import click
import shutil
import re
import json
import tempfile
import torch
import gc
from omegaconf import OmegaConf, DictConfig

from src import dnnlib
from training import training_loop
from metrics import metric_main
from src.torch_utils import training_stats
from src.torch_utils import custom_ops

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

augpipe_specs = {
    'blit':      dict(xflip=1, rotate90=1, xint=1),
    'geom':      dict(scale=1, rotate=1, aniso=1, xfrac=1),
    'color':     dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
    'filter':    dict(imgfilter=1),
    'noise':     dict(noise=1),
    'cutout':    dict(cutout=1),
    'bg':        dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
    'bgc':       dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
    'bgcf':      dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
    'bgcfn':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
    'bgcfnc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    'easy':      dict(xflip=1, xint=1, scale=1, rotate=0.5, rotate_max=0.1, xfrac=1, noise=0.1, cutout=1, cutout_size=0.25),
    'bgc_norgb': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, cutout=1),
}

#----------------------------------------------------------------------------

def process_hyperparams(cfg: DictConfig):
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------

    c = cfg.training
    assert isinstance(c.gpus, int)
    if not (c.gpus >= 1 and c.gpus & (c.gpus - 1) == 0):
        raise UserError('`gpus` must be a power of two')
    args.num_gpus = c.gpus

    if c.snap is None:
        c.snap = 50
    assert isinstance(c.snap, int)
    if c.snap < 1:
        raise UserError('`snap` must be at least 1')
    args.image_snapshot_ticks = c.snap
    args.network_snapshot_ticks = c.snap

    if not all(metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise UserError('\n'.join(['`metrics` can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = [m for m in c.metrics]

    assert isinstance(c.seed, int)
    args.random_seed = c.seed

    # -----------------------------------
    # Dataset: data, cond, subset, mirror
    # -----------------------------------

    assert c.data is not None
    assert isinstance(c.data, str)
    assert cfg.model.loss_kwargs.style_mixing_prob == 0, "Not supported"

    args.training_set_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.VideoFramesFolderDataset',
        path=c.data, cfg=cfg.dataset, use_labels=True, max_size=None, xflip=False)

    # args.training_set_kwargs = dnnlib.EasyDict(
    #     class_name='training.dataset.ImageFolderDataset',
    #     path=data, use_labels=True, max_size=None, xflip=False)

    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=c.num_workers, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset

        args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        args.training_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
        args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
        desc = training_set.name
        del training_set; gc.collect() # conserve memory
    except IOError as err:
        raise UserError(f'--data: {err}')

    assert cfg.dataset.c_dim >= 0
    if cfg.dataset.c_dim > 0:
        if not args.training_set_kwargs.use_labels:
            raise UserError('Conditional training requires labels specified in dataset.json')
    else:
        args.training_set_kwargs.use_labels = False

    if c.subset is not None:
        assert isinstance(c.subset, int)
        if not 1 <= c.subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--c.subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-c.subset{c.subset}'
        if c.subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = c.subset
            args.training_set_kwargs.random_seed = args.random_seed

    assert isinstance(c.mirror, bool)
    if c.mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch_size
    # ------------------------------------

    assert isinstance(c.cfg, str)
    desc += f'-{c.cfg}'

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     r1_gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  r1_gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, r1_gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, r1_gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  r1_gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, r1_gamma=0.01, ema=500, ramp=0.05, map=2),
    }

    assert c.cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[c.cfg])
    if c.cfg == 'auto':
        desc += f'{c.gpus:d}'
        spec.ref_gpus = c.gpus
        res = args.training_set_kwargs.resolution
        if c.batch_size is None:
            spec.mb = max(min(c.gpus * min(4096 // res, 32), 64), c.gpus) # keep gpu memory consumption at bay
        else:
            spec.mb = c.batch_size
        spec.mbstd = min(spec.mb // c.gpus, cfg.model.discriminator.mbstd_group_size) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.r1_gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32

    args.G_kwargs = dnnlib.EasyDict(
        class_name=f'training.{cfg.model.generator.source}.Generator',
        w_dim=cfg.model.generator.w_dim, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name=f'training.{cfg.model.discriminator.source}.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = int(cfg.model.generator.get('fmaps', spec.fmaps) * 32768)
    args.D_kwargs.channel_base = int(cfg.model.discriminator.get('fmaps', spec.fmaps) * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = cfg.model.generator.get('channel_max', 512)
    args.D_kwargs.channel_max = cfg.model.discriminator.get('channel_max', 512)
    args.G_kwargs.mapping_kwargs.num_layers = cfg.model.generator.get('mapping_net_n_layers', spec.map)
    args.G_kwargs.mapping_kwargs.cfg = cfg.model.generator
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.G_kwargs.cfg = cfg.model.generator
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd
    args.D_kwargs.cfg = cfg.model.discriminator
    args.D_kwargs.mapping_kwargs.num_layers = 2

    if cfg.model.generator.get('fp32', False):
        args.G_kwargs.synthesis_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = None

    if cfg.model.discriminator.get('fp32', False):
        args.D_kwargs.num_fp16_res = 0
        args.D_kwargs.conv_clamp = None

    G_lr = cfg.model.optim.generator.get('lr', spec.lrate)
    D_lr = cfg.model.optim.discriminator.get('lr', spec.lrate)
    G_betas = cfg.model.optim.generator.get('betas', [0, 0.99])
    D_betas = cfg.model.optim.discriminator.get('betas', [0, 0.99])
    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=G_lr, betas=G_betas, eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=D_lr, betas=D_betas, eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name=f'training.loss.{cfg.model.loss_kwargs.source}', r1_gamma=spec.r1_gamma, cfg=cfg)
    args.cfg = cfg

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if cfg == 'cifar':
        args.loss_kwargs.pl_weight = 0 # disable path length regularization
        args.loss_kwargs.style_mixing_prob = 0 # disable style mixing
        args.D_kwargs.architecture = 'orig' # disable residual skip connections

    if 'r1_gamma' in cfg.model.loss_kwargs:
        r1_gamma = cfg.model.loss_kwargs.r1_gamma
        assert isinstance(r1_gamma, float)
        if not r1_gamma >= 0:
            raise UserError('r1_gamma must be non-negative')
        desc += f'-r1_gamma{r1_gamma:g}'
        args.loss_kwargs.r1_gamma = r1_gamma

    if 'style_mixing_prob' in cfg.model.loss_kwargs:
        args.loss_kwargs.style_mixing_prob = cfg.model.loss_kwargs.style_mixing_prob

    if 'pl_weight' in cfg.model.loss_kwargs:
        args.loss_kwargs.pl_weight = cfg.model.loss_kwargs.pl_weight

    if c.kimg is not None:
        assert isinstance(c.kimg, int)
        if not c.kimg >= 1:
            raise UserError('--c.kimg must be at least 1')
        desc += f'-kimg{c.kimg:d}'
        args.total_kimg = c.kimg

    if c.batch_size is not None:
        assert isinstance(c.batch_size, int)
        if not (c.batch_size >= 1 and c.batch_size % c.gpus == 0):
            raise UserError('--c.batch_size must be at least 1 and divisible by --gpus')
        desc += f'-batch_size{c.batch_size}'
        args.batch_size = c.batch_size
        args.batch_gpu = c.batch_size // c.gpus

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------

    assert isinstance(c.aug, str)
    desc += f'-{c.aug}'

    if c.aug == 'ada':
        args.ada_target = 0.6
    elif c.aug == 'noaug':
        pass
    elif c.aug == 'fixed':
        if c.p is None:
            raise UserError(f'--aug={c.aug} requires specifying --p')

    else:
        raise UserError(f'--aug={c.aug} not supported')

    if c.p is not None:
        assert isinstance(c.p, float)
        if c.aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= c.p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc += f'-p{c.p:g}'
        args.augment_p = c.p

    if c.target is not None:
        assert isinstance(c.target, float)
        if c.aug != 'ada':
            raise UserError('--target can only be specified with --aug=ada')
        if not 0 <= c.target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{c.target:g}'
        args.ada_target = c.target

    assert c.augpipe is None or c.augpipe in augpipe_specs
    if not c.augpipe is None:
        assert c.aug != 'noaug', '--augpipe cannot be specified with --aug=noaug'
    if c.aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[c.augpipe])

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    assert c.resume is None or isinstance(c.resume, str)
    args.resume_whole_state = False
    if c.resume is None or c.resume == 'noresume':
        pass
    elif c.resume in resume_specs:
        desc += f'-resume{c.resume}'
        args.resume_pkl = resume_specs[c.resume] # predefined url
    elif c.resume == 'latest':
        ckpt_regex = re.compile("^network-snapshot-\d{6}.pkl$")
        run_dir = os.path.join(cfg.project_release_dir, 'output')
        ckpts = sorted([f for f in os.listdir(run_dir) if ckpt_regex.match(f)]) if os.path.isdir(run_dir) else []

        if len(ckpts) > 0:
            args.resume_pkl = os.path.join(run_dir, ckpts[-1])
            args.resume_whole_state = True
            desc += f'-resume-latest-{ckpts[-1]}'
        else:
            print("Was requested to resume training from the latest checkpoint, but no checkpoints found.")
            print('So will start from scratch.')
            desc += '-resume-latest-not-found'
    else:
        desc += '-resumecustom'
        args.resume_pkl = c.resume # custom path or url

    if not c.resume in {'noresume', 'latest'}:
        args.ada_kimg = 100 # make ADA react faster at the beginning
        args.ema_rampup = None # disable EMA rampup

    if c.freezed is not None:
        assert isinstance(c.freezed, int)
        if not c.freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{c.freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = c.freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    assert isinstance(c.fp32, bool)
    if c.fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    assert isinstance(c.nhwc, bool)
    if c.nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    assert isinstance(c.nobench, bool)
    if c.nobench:
        args.cudnn_benchmark = False

    assert isinstance(c.allow_tf32, bool)
    if c.allow_tf32:
        args.allow_tf32 = True

    assert isinstance(c.num_workers, int)
    assert c.num_workers >= 1, '`num_workers` must be at least 1'
    args.data_loader_kwargs.num_workers = c.num_workers

    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **args)

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

def main():
    dnnlib.util.Logger(should_flush=True)
    cfg = OmegaConf.load("experiment_config.yaml")
    OmegaConf.set_struct(cfg, True)

    # Setup training options.
    _run_desc, args = process_hyperparams(cfg)

    # Pick output directory.
    args.run_dir = os.path.join(cfg.training.outdir, 'output')

    # Print options.
    print()
    print('Training config is located in `experiment_config.yaml`')
    print()
    if cfg.env.get('symlink_output', None):
        print(f'Output directory:   {args.run_dir} (symlinked to {cfg.env.symlink_output})')
    else:
        print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {args.training_set_kwargs.path}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print(f'Number of videos:   {args.training_set_kwargs.max_size}')
    print(f'Image resolution:   {args.training_set_kwargs.resolution}')
    print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if cfg.training.dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    if cfg.env.get('symlink_output', None):
        if os.path.exists(cfg.env.symlink_output) and not args.resume_whole_state:
            print(f'Deleting old output dir: {cfg.env.symlink_output} ...')
            shutil.rmtree(cfg.env.symlink_output)

        if not args.resume_whole_state:
            os.makedirs(cfg.env.symlink_output, exist_ok=False)
            os.symlink(cfg.env.symlink_output, args.run_dir)
            print(f'Symlinked `output` into `{cfg.env.symlink_output}`')
        else:
            print(f'Did not symlink `{cfg.env.symlink_output}` since resuming training.')
    else:
        os.makedirs(args.run_dir, exist_ok=args.resume_whole_state)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
