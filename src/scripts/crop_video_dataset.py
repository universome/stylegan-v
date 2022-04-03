import os
import shutil
import argparse
from typing import List

import numpy as np
from tqdm import tqdm
from PIL import Image


def crop_video_dataset(source_dir: str, max_num_frames: int=None, slice_n_left_frames: int=0, resize: int=None, target_dir: str=None):
    dataset_name = os.path.basename(source_dir)
    if target_dir is None:
        max_num_frames_prefix = "" if max_num_frames is None else f"_cut{max_num_frames}"
        slice_prefix = "" if slice_n_left_frames == 0 else f"_slice{slice_n_left_frames}"
        new_dataset_name = f"{dataset_name}{max_num_frames_prefix}{slice_prefix}"
        target_dir = os.path.join(os.path.dirname(source_dir), new_dataset_name)
    all_clips_paths = listdir_full_paths(source_dir)
    os.makedirs(target_dir, exist_ok=True)
    slice_proportions = []

    total_num_frames = 0

    for source_clip_dir in tqdm(all_clips_paths, desc=f'Cropping the dataset into {target_dir}'):
        all_frames = listdir_full_paths(source_clip_dir)
        if len(all_frames) == 0:
            print(f'{source_clip_dir} is empty. Skipping it.')
            continue
        target_clip_dir = os.path.join(target_dir, os.path.basename(source_clip_dir))
        os.makedirs(target_clip_dir, exist_ok=True)
        total_num_frames += len(all_frames)
        slice_proportions.append(slice_n_left_frames / len(all_frames))
        all_frames = all_frames[slice_n_left_frames:]

        if not max_num_frames is None:
            all_frames = all_frames[:max_num_frames]

        for source_frame_path in all_frames:
            target_frame_path = os.path.join(target_clip_dir, os.path.basename(source_frame_path))

            if resize is None:
                shutil.copy(source_frame_path, target_frame_path)
            else:
                assert target_frame_path.endswith('.jpg')
                Image.open(source_frame_path).resize((resize, resize), resample=Image.LANCZOS).save(target_frame_path, q=95)

    print(f'Done! Sliced {np.mean(slice_proportions) * 100.0 : .02f}% on average. {len(all_clips_paths) * slice_n_left_frames / total_num_frames * 100.0 : .02f}% of total num frames.')


def listdir_full_paths(d) -> List[os.PathLike]:
    return sorted([os.path.join(d, x) for x in os.listdir(d)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crops a video dataset temporally into several frames')
    parser.add_argument('source_dir', type=str, help='Path to the dataset')
    parser.add_argument('-n', '--max_num_frames', type=int, default=None, help='Number of frames to preserve')
    parser.add_argument('--slice_n_left_frames', type=int, default=0, help='Number of frames to slice from the left')
    parser.add_argument('--resize', type=int, default=None, help='Should we resize the dataset')
    parser.add_argument('--target_dir', type=str, default=None, help='Should we resize the dataset')
    args = parser.parse_args()

    crop_video_dataset(
        source_dir=args.source_dir,
        max_num_frames=args.max_num_frames,
        slice_n_left_frames=args.slice_n_left_frames,
        resize=args.resize,
        target_dir=args.target_dir,
    )
