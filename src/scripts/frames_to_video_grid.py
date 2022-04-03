"""
Converts a directory of video frames into an mp4-grid
"""
import sys; sys.path.extend(['.'])
import os
import argparse
import random

import numpy as np
import torch
from torch import Tensor
import torchvision.transforms.functional as TVF
from torchvision import utils
from PIL import Image
from tqdm import tqdm
import torchvision


def frames_to_video_grid(videos_dir: os.PathLike, num_videos: int, length: int, fps: int, output_path: os.PathLike, select_random: bool=False, random_seed: int=None):
    clips_paths = [os.path.join(videos_dir, d) for d in os.listdir(videos_dir)]

    # bad_idx = [0, 9, 11, 16]
    # clips_paths = [c for i, c in enumerate(clips_paths) if not i in bad_idx]

    if select_random:
        random.seed(random_seed)
        clips_paths = random.sample(clips_paths, k=num_videos)
    else:
        clips_paths = clips_paths[:num_videos]
    videos = [read_first_n_frames(d, length) for d in tqdm(clips_paths, desc='Reading data...')] # [num_videos, length, c, h, w]
    videos = [fill_with_black_squares(v, length) for v in tqdm(videos, desc='Adding empty frames')] # [num_videos, length, c, h, w]
    frame_grids = torch.stack(videos).permute(1, 0, 2, 3, 4) # [video_len, num_videos, c, h, w]
    frame_grids = [utils.make_grid(fs, nrow=int(np.ceil(np.sqrt(num_videos)))) for fs in tqdm(frame_grids, desc='Making grids')]

    if os.path.dirname(output_path) != "":
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frame_grids = (torch.stack(frame_grids) * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, H, W, C]
    torchvision.io.write_video(output_path, frame_grids, fps=fps, video_codec='h264', options={'crf': '10'})


def read_first_n_frames(d: os.PathLike, num_frames: int) -> Tensor:
    images = [Image.open(os.path.join(d, f)) for f in sorted(os.listdir(d))[:num_frames]]
    images = [TVF.to_tensor(x) for x in images]

    return torch.stack(images)


def fill_with_black_squares(video, desired_len: int) -> Tensor:
    if len(video) >= desired_len:
        return video

    return torch.cat([
        video,
        torch.zeros_like(video[0]).unsqueeze(0).repeat(desired_len - len(video), 1, 1, 1),
    ], dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, help='Directory with video frames')
    parser.add_argument('-n', '--num_videos', type=int, help='Number of videos to consider')
    parser.add_argument('-l', '--length', type=int, help='Video length (in frames)')
    parser.add_argument('--fps', type=int, default=25, help='FPS to save with.')
    parser.add_argument('-o', '--output_path', type=str, help='Where to save the file?.')
    parser.add_argument('--select_random', action='store_true', help='Select videos at random?')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed when selecting videos at random')

    args = parser.parse_args()

    frames_to_video_grid(
        videos_dir=args.directory,
        num_videos=args.num_videos,
        length=args.length,
        fps=args.fps,
        output_path=args.output_path,
        select_random=args.select_random,
        random_seed=args.random_seed,
    )
