"""
Converts a dataset of mp4 videos into a dataset of video frames
I.e. a directory of mp4 files becomes a directory of directories of frames
This speeds up loading during training because we do not need
"""
import os
from typing import List
import argparse
from pathlib import Path
from multiprocessing import Pool
from collections import Counter

import numpy as np
from PIL import Image
import torchvision.transforms.functional as TVF
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def convert_videos_into_dataset(video_path: os.PathLike, target_dir: os.PathLike, num_chunks: int, chunk_size: int, start_frame: int, target_size: int, force_fps: int):
    assert (num_chunks is None) or (chunk_size is None), "Cant use both num_chunks and chunk_size"

    os.makedirs(target_dir, exist_ok=True)
    clip = VideoFileClip(video_path)
    fps = clip.fps if force_fps is None else force_fps
    num_frames_total = int(np.floor(clip.duration * fps)) - start_frame

    if num_chunks is None:
        num_chunks = num_frames_total // chunk_size
    else:
        chunk_size = num_frames_total // num_chunks

    num_frames_to_save = chunk_size * num_chunks

    print(f'Processing the video at {fps} fps. {num_frames_total} frames in total. We have {num_chunks} videos of {chunk_size} frames each.')

    current_chunk_idx = 0
    frame_idx = -start_frame
    curr_chunk_dir = os.path.join(target_dir, f'{current_chunk_idx:06d}')

    for frame in tqdm(clip.iter_frames(fps=fps), total=num_frames_total + start_frame):
        if frame_idx >= 0:
            os.makedirs(curr_chunk_dir, exist_ok=True)
            frame = Image.fromarray(frame)
            frame = TVF.center_crop(frame, output_size=min(frame.size))
            frame = TVF.resize(frame, size=target_size, interpolation=Image.LANCZOS)
            frame.save(os.path.join(curr_chunk_dir, f'{frame_idx % chunk_size:06d}.jpg'), q=95)

        frame_idx += 1
        if frame_idx % chunk_size == 0 and frame_idx > 0:
            current_chunk_idx += 1
            curr_chunk_dir = os.path.join(target_dir, f'{current_chunk_idx:06d}')

        if frame_idx == num_frames_to_save:
            # Stop here so not to have a partially-filled chunk
            break

    chunk_sizes = [len(os.listdir(d)) for d in listdir_full_paths(target_dir)]
    assert len(set(chunk_sizes)) == 1, f"Bad chunk sizes: {set(chunk_sizes)}"

    print('Finished successfully!')


def listdir_full_paths(d) -> List[os.PathLike]:
    return sorted([os.path.join(d, x) for x in os.listdir(d)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a long video into a dataset of frame dirs')
    parser.add_argument('-s', '--source_video_path', type=str, help='Path to the source video')
    parser.add_argument('-t', '--target_dir', type=str, help='Where to save the new dataset')
    parser.add_argument('-n', '--num_chunks', type=int, help='How many samples should there be in the dataset?')
    parser.add_argument('-cs', '--chunk_size', type=int, help='Each video length. Should be used separately from num_chunks')
    parser.add_argument('-sf', '--start_frame', type=int, default=0, help='Start frame idx. Should we skip several frames?')
    parser.add_argument('--target_size', type=int, default=128, help='What size should we resize to?')
    parser.add_argument('--force_fps', type=int, help='What fps should we run videos with?')
    args = parser.parse_args()

    convert_videos_into_dataset(
        video_path=args.source_video_path,
        target_dir=args.target_dir,
        num_chunks=args.num_chunks,
        chunk_size=args.chunk_size,
        start_frame=args.start_frame,
        target_size=args.target_size,
        force_fps=args.force_fps,
    )
