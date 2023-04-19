# StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2
### [CVPR 2022] Official pytorch implementation
[[Project website]](https://universome.github.io/stylegan-v)
[[Paper]](https://kaust-cair.s3.amazonaws.com/stylegan-v/stylegan-v-paper.pdf)
[[Casual GAN papers summary]](https://www.casualganpapers.com/text_guided_video_editing_hd_video_generation/StyleGAN-V-explained.html?query=stylegan-v)

<div style="text-align:center">
<img src="https://user-images.githubusercontent.com/3128824/161441271-09fa5cfe-a2ae-4a7f-b5ca-ad90f5e0287e.gif" alt="Content/Motion decomposition for Face Forensics 256x256"/>
</div>

<div style="text-align:center">
<img src="https://user-images.githubusercontent.com/3128824/161441278-c7c3a43d-a3cd-417b-98c5-6b889ac32935.gif" alt="Content/Motion decomposition for Sky Timelapse 256x256"/>
</div>

Code release TODO:
- [x] Installation guide
- [x] Training code
- [x] Data preprocessing scripts
- [ ] CLIP editing scripts (50% done)
- [ ] Jupyter notebook demos
- [x] [Pre-trained checkpoints](https://disk.yandex.ru/d/v7MS7zu4mmZxXw)

## Installation
To install and activate the environment, run the following command:
```
conda env create -f environment.yaml -p env
conda activate ./env
```
For clip editing, you will need to install [StyleCLIP](https://github.com/orpatashnik/StyleCLIP) and `clip`.
This repo is built on top of [INR-GAN](https://github.com/universome/inr-gan), so make sure that it runs on your system.

If you have Ampere GPUs (A6000, A100 or RTX-3090), then use `environment-ampere.yaml` instead because it is based CUDA 11 and newer pytorch versions.

## System requirements

Our codebase uses the same system requirements as StyleGAN2-ADA: see them [here](https://github.com/NVlabs/stylegan2-ada-pytorch#requirements).
We trained all the 256x256 models on 4 V100s with 32 GB each for ~2 days.
It is very similar in training time to [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) (even a bit faster).

## Training
### Dataset structure
The dataset should be either a `.zip` archive (the default setting) or a directory structured as:
```
dataset/
    video1/
        - frame1.jpg
        - frame2.jpg
        - ...
    video2/
        - frame1.jpg
        - frame2.jpg
        - ...
    ...
```
We use such frame-wise structure because it makes loading faster for sparse training.

By default, we assume that the data is packed into a `.zip` archive since such representation is useful to avoid additional overhead when copying data between machines on a cluster.
You can also train from a directory: for this, just remove the `.zip` suffix from the `dataset.path` property in `configs/dataset/base.yaml`.

If  you want to train on a custom dataset, then create a config for it here `configs/dataset/my_dataset_config_name.yaml` (see `configs/dataset/ffs.yaml` as an example).
The `fps` parameter is needed for visualizations purposes only, videos typically have the value of 25 or 30 FPS.

### Training StyleGAN-V
To train on FaceForensics 256x256, run:
```
python src/infra/launch.py hydra.run.dir=. exp_suffix=my_experiment_name env=local dataset=ffs dataset.resolution=256 num_gpus=4
```

To train on SkyTimelapse 256x256, run:
```
python src/infra/launch.py hydra.run.dir=. exp_suffix=my_experiment_name env=local dataset=sky_timelapse dataset.resolution=256 num_gpus=4 model.generator.time_enc.min_period_len=256
```
For SkyTimelapse 256x256, we increased the period length for the motion time encoder since the motions in this dataset are much slower/smoother, than in FaceForensics.
In practice, this parameter (and its accompanying `model.generator.motion.motion_z_distance`) influences the motion quality (but not the image quality!) the most.

If you do not want `hydra` to create some log directories (typically, you don't), add the following arguments: `hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled`.

In case [slurm](https://slurm.schedmd.com/documentation.html) is installed on your system, you can submit the slurm job with the above training by adding `slurm=true` parameter.
Sbatch arguments are specified in `configs/infra.yaml`, you can update them with your required ones.
Also note that you can create your own environment in `configs/env`.

On older GPUs (non V100 and newer), custom CUDA kernels (bias_act and upfirdn2n) might fail to compile. The following two lines can help:
```
export TORCH_CUDA_ARCH_LIST="7.0"
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
```

### Resume training
If you shut down your experiment at some point and would love to fully recover training (i.e., with the optimizer parameters, logging, etc.), the add `training.resume=latest` argument to your launch script, e.g.:
```
python src/infra/launch.py hydra.run.dir=. exp_suffix=my_experiment_name env=local dataset=ffs dataset.resolution=256 num_gpus=4 training.resume=latest
```
It will locate the given experiment directory (note that the git hash and the `exp_suffix` must be the same) and resume the training from it.

### Inference
To sample from the model, launch the following command:
```
python src/scripts/generate.py --network_pkl /path/to/network-snapshot.pkl --num_videos 25 --as_grids true --save_as_mp4 true --fps 25 --video_len 128 --batch_size 25 --outdir /path/to/output/dir --truncation_psi 0.9
```
This will sample 25 videos of 25 FPS as a 5x5 grid with the truncation factor of 0.9.
Each video consists of 128 frames.
Adjust the corresponding arguments to change the settings.

Alternatively, instead of specifying `--network_pkl`, you can specify `--networks_dir`, which should lead to a directory containing the checkpoints and the `metric-fvd2048_16f.json` metrics json file (it is generated automatically during training).
It will then select the best checkpoint based on the metrics, which so not to search for the best checkpoint of an experiment manually.

To sample content/motion decomposition grids, use `--moco_decomposition 1` by running the following command:
```
python src/scripts/generate.py --networks_dir PATH_TO_EXPERIMENT/output --num_videos 25 --as_grids true --save_as_mp4 true --fps 25 --video_len 128 --batch_size 25 --outdir tmp --truncation_psi 0.8 --moco_decomposition 1
```

### Training MoCoGAN + SG2 backbone
To train the `MoCoGAN+SG2` model, just use the `mocogan.yaml` model config with the uniform sampling:
```
python src/infra/launch.py hydra.run.dir=. +exp_suffix=my_experiment env=local dataset=sky_timelapse dataset.resolution=256 num_gpus=4 model=mocogan sampling=uniform sampling.max_dist_between_frames=1
```

### Training other baselines
To train other baselines, used in the paper, we used their original implementations:
- [MoCoGAN](https://github.com/sergeytulyakov/mocogan)
- [MoCoGAN-HD](https://github.com/snap-research/MoCoGAN-HD)
- [DIGAN](https://github.com/sihyun-yu/digan)
- [VideoGPT](https://github.com/wilson1yan/VideoGPT)

## Data
Datasets can be downloaded here:
- SkyTimelapse: https://github.com/weixiong-ur/mdgan
- UCF: https://www.crcv.ucf.edu/data/UCF101.php
- FaceForensics: https://github.com/ondyari/FaceForensics
- RainbowJelly: https://www.youtube.com/watch?v=P8Bit37hlsQ
- MEAD: https://wywu.github.io/projects/MEAD/MEAD.html

We resize all the datasets to the 256x256 resolution (except for MEAD, which we resize to 1024x1024).
FFS was preprocessed with `src/scripts/preprocess_ffs.py` to extract face crops.
For MEAD, we used only the front views.

For `RainbowJelly`, download the youtube video, save it as `rainbow_jelly.mp4` and convert into the dataset by running:
```
python src/scripts/convert_video_to_dataset.py -s /path/to/rainbow_jelly.mp4 -t /path/to/desired/directory --target_size 256 -sf 150 -cs 512
```

## Evaluation
In this repo, we re-implemented two popular evaluation measures for video generation:
- [Frechet Video Distance](https://arxiv.org/abs/1812.01717). For this, we re-implemented *perfectly* (up to numerical precision) the original Tensorflow version of the I3D model trained on Kinetics-400 and converted it to TorchScript. This is a precise implementation of the official one and we set up [this comparison repo](https://github.com/universome/fvd-comparison) to demonstrate this.
- [Inception Score](https://arxiv.org/abs/1611.06624) (used only for UCF101). For this, we re-implemented *perfectly* (up to numerical precision) the original [Chainer version of the UCF101-finetuned C3D model](https://github.com/pfnet-research/tgan2) in Pytorch and converted it to TorchScript.

In practice, we found that neither Frechet Video Distance nor Inception Score work well reliably for catching motion artifacts.
This creates the need for better evaluation measures.

Advantages of our metrics implementation compared to the original ones:
- It is much faster due to TorchScript and parallelization across several GPUs
- It can be launched both on top of both a generator checkpoint and off-the-shelf samples
- It is implemented in a very recent Pytorch version (v1.9.0) instead of deprecated TensorFlow 1.14 or Chainer 6.0
- It is directly incorporated into training to track progress online without the need to launch the evaluation separately
- For FVD, our implementation is *complete*, while the original one provides evaluation for a single batch of already processed videos only
- Our FVD implementation supports different subsampling strategies and a variable number of frames in a video.

To compute FVD between two datasets, run the following command:
```
python src/scripts/calc_metrics_for_dataset.py --real_data_path /path/to/dataset_a.zip --fake_data_path /path/to/dataset_b.zip --mirror 1 --gpus 4 --resolution 256 --metrics fvd2048_16f,fvd2048_128f,fvd2048_128f_subsample8f,fid50k_full --verbose 0 --use_cache 0
```

To compute FVD for a trained model, run `src/scripts/calc_metrics.py` instead.

Both datasets should be in the format specified above.
They can be either zip archives or normal directories.
This will compute several metrics:
- `fid50k_full` - Frechet Inception Distance
- `fvd2048_16f` — Frechet Video Distance with 16 frames
- `fvd2048_128f` - Frechet Video Distance with 128 frames
- `fvd2048_128f_subsample8f` — Frechet Video Distance with 16 frames, but sampled with a 8-frames interval

*Note*. If you face any trouble running the above evaluation scripts — please do not hesitate contacting us!

## Projection and CLIP editing
This section is still under construction.
We will update it shortly.

Those two files provide projection and editing scripts:
- `src/scripts/project.py`
- `src/scripts/clip_edit.py`

## Infrastructure and visualization
You will find some useful scripts for data processing and visualization in `src/scripts`

## Troubleshooting
Make sure that [INR-GAN](https://github.com/universome/inr-gan) and [StyleGAN2-ADA](https://github.com/nvlabs/stylegan2-ada) are runnable on your system.
We do not use any additional CUDA kernels or any exotic dependencies.

If this didn't help, than it's likely there is some dependency version mismatch.
Check the versions of your installed dependencies with:
```
pip freeze
```
and compare them with the ones specified in `environment.yaml`/`environment-ampere.yaml`.

If this didn't help — open an issue, it's likely that the problem is on our side.

## License
This repo is built on top of [INR-GAN](https://github.com/universome/inr-gan), which is likely to be restricted by the [NVidia license](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html) since it's built on top of [StyleGAN2-ADA](https://github.com/nvlabs/stylegan2-ada).
If that's the case, then this repo is also restricted by it.


## Bibtex
```
@misc{stylegan_v,
    title={StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2},
    author={Ivan Skorokhodov and Sergey Tulyakov and Mohamed Elhoseiny},
    journal={arXiv preprint arXiv:2112.14683},
    year={2021}
}

@inproceedings{digan,
    title={Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks},
    author={Sihyun Yu and Jihoon Tack and Sangwoo Mo and Hyunsu Kim and Junho Kim and Jung-Woo Ha and Jinwoo Shin},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=Czsdv-S4-w9}
}
```
