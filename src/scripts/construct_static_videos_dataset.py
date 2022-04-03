"""
Takes a dataset directory and repeats the frames to include only a random frame from each video
This is needed to calculate same-frame FVD and DiFID
"""
import os
import random
import argparse
from typing import List
import shutil
from tqdm import tqdm


def construct_static_videos_dataset(videos_dir: os.PathLike, max_len: int=None, output_dir: os.PathLike=None, force_len: int=None):
    output_dir = output_dir if not output_dir is None else f'{videos_dir}_freeze'
    clips_paths = [os.path.join(videos_dir, d) for d in os.listdir(videos_dir)]

    print(f'Saving into {output_dir}')

    for video_idx, clip_path in enumerate(tqdm(clips_paths)):
        frames_paths = os.listdir(clip_path)
        frame_to_repeat = random.choice(frames_paths)
        curr_output_dir = os.path.join(output_dir, f'{video_idx:05d}')
        os.makedirs(curr_output_dir, exist_ok=True)
        num_frames_to_create = force_len if not force_len is None else min(len(frames_paths), max_len)

        for i in range(num_frames_to_create):
            ext = os.path.splitext(frame_to_repeat)[1].lower()
            target_file_path = os.path.join(curr_output_dir, f'{i:06d}{ext}')
            shutil.copy(os.path.join(clip_path, frame_to_repeat), target_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, help='Directory with video frames')
    parser.add_argument('-o', '--output_dir', type=None, help='Where to save the file?.')
    parser.add_argument('-l', '--max_len', type=int, help='Max video length')
    parser.add_argument('-fl', '--force_len', type=int, help='Force video length')

    args = parser.parse_args()

    construct_static_videos_dataset(
        videos_dir=args.directory,
        max_len=args.max_len,
        output_dir=args.output_dir,
        force_len=args.force_len,
    )
