"""
This file preprocesses FaceForensics dataset by cropping it
Copied from https://github.com/pfnet-research/tgan2/blob/master/scripts/make_face_forensics.py
"""

import argparse
import os
from typing import List
from multiprocessing import Pool
from PIL import Image

import cv2
# import h5py
import imageio
import numpy as np
import pandas
from tqdm import tqdm


def parse_videos(source_dir, splits: List[str], categories: List[dir]):
    results = []
    for split in splits:
        for category in categories:
            target_dir = os.path.join(source_dir, split, category)
            filenames = sorted(os.listdir(target_dir))
            for filename in filenames:
                results.append({
                    'split': split,
                    'category': category,
                    'filename': filename,
                    'filepath': os.path.join(split, category, filename),
                })
    return pandas.DataFrame(results)


def crop(img, left, right, top, bottom, margin):
    cols = right - left
    rows = bottom - top
    if cols < rows:
        padding = rows - cols
        left -= padding // 2
        right += (padding // 2) + (padding % 2)
        cols = right - left
    else:
        padding = cols - rows
        top -= padding // 2
        bottom += (padding // 2) + (padding % 2)
        rows = bottom - top
    assert(rows == cols)
    return img[top:bottom, left:right]


def job_proxy(kwargs):
    process_and_save_video(**kwargs)


def process_and_save_video(video_path: os.PathLike, mask_path: os.PathLike, img_size: int, wide_crop: bool, output_dir: os.PathLike):
    try:
        video = process_video(video_path, mask_path, img_size=img_size, wide_crop=wide_crop)
    except KeyboardInterrupt:
        raise
    except:
        print(f'Couldnt process {video_path}')
        return

    os.makedirs(output_dir, exist_ok=True)

    # if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
    #     return

    for i, frame in enumerate(video):
        frame = frame.transpose(1, 2, 0)
        Image.fromarray(frame).save(os.path.join(output_dir, f'{i:06d}.jpg'), q=95)


def process_video(video_path, mask_path, img_size, threshold=5, margin=0.02, wide_crop: bool=False):
    video_reader = imageio.get_reader(video_path)
    mask_reader = imageio.get_reader(mask_path)
    assert(video_reader.get_length() == mask_reader.get_length())

    # Searching for the widest crop which would work for the whole video
    if wide_crop:
        left_most = float('inf')
        top_most = float('inf')
        right_most = float('-inf')
        bottom_most = float('-inf')

        for img, mask in zip(video_reader, mask_reader):
            hist = (255 - mask).astype(np.float64).sum(axis=2)
            horiz_hist = np.where(hist.mean(axis=0) > threshold)[0]
            vert_hist = np.where(hist.mean(axis=1) > threshold)[0]
            left, right = horiz_hist[0], horiz_hist[-1]
            top, bottom = vert_hist[0], vert_hist[-1]
            left_most = min(left_most, left)
            top_most = min(top_most, top)
            right_most = max(right_most, right)
            bottom_most = max(bottom_most, bottom)

    video = []
    for img, mask in zip(video_reader, mask_reader):
        if wide_crop:
            left, right, top, bottom = left_most, right_most, top_most, bottom_most
        else:
            hist = (255 - mask).astype(np.float64).sum(axis=2)
            horiz_hist = np.where(hist.mean(axis=0) > threshold)[0]
            vert_hist = np.where(hist.mean(axis=1) > threshold)[0]
            left, right = horiz_hist[0], horiz_hist[-1]
            top, bottom = vert_hist[0], vert_hist[-1]

        dst_img = crop(img, left, right, top, bottom, margin)

        try:
            dst_img = cv2.resize(
                dst_img, (img_size, img_size),
                interpolation=cv2.INTER_LANCZOS4).transpose(2, 0, 1)
            video.append(dst_img)
        except KeyboardInterrupt:
            raise
        except:
            print(img.shape, dst_img.shape, left, right, top, bottom)

    T = len(video)
    video = np.concatenate(video).reshape(T, 3, img_size, img_size)
    return video


# def count_frames(path):
#     reader = imageio.get_reader(path)
#     n_frames = 0
#     while True:
#         try:
#             img = reader.get_next_data()
#         except IndexError as e:
#             break
#         else:
#             n_frames += 1
#     return n_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='data/FaceForensics_compressed')
    parser.add_argument('--output_dir', type=str, default='data/ffs_processed')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--wide_crop', action='store_true', help="Should we crop each frame independently (this makes a video shaking)?")
    args = parser.parse_args()

    # splits = ['train', 'val', 'test']
    # categories = ['original', 'mask', 'altered']
    splits = ['train']
    categories = ['original', 'mask']
    df = parse_videos(args.source_dir, splits, categories)
    os.makedirs(args.output_dir, exist_ok=True)

    for split in splits:
        target_frame = df[df['split'] == split]
        filenames = target_frame['filename'].unique()

        # print('Count # of frames')
        # rets = []
        # for i, filename in enumerate(filenames):
        #     fn_frame = target_frame[target_frame['filename'] == filename]
        #     video_path = os.path.join(
        #         args.source_dir, fn_frame[fn_frame['category'] == 'original'].iloc[0]['filepath'])
        #     rets.append(p.apply_async(count_frames, args=(video_path,)))
        # n_frames = 0
        # for ret in tqdm(rets):
        #     n_frames += ret.get()
        # print('# of frames: {}'.format(n_frames))

        # h5file = h5py.File(os.path.join(args.output_dir, '{}.h5'.format(split)), 'w')
        # dset = h5file.create_dataset('image', (n_frames, 3, args.img_size, args.img_size), dtype=np.uint8)
        # conf = []
        # start = 0

        pool = Pool(processes=args.num_workers)
        job_kwargs_list = []

        for i, filename in enumerate(filenames):
            fn_frame = target_frame[target_frame['filename'] == filename]
            video_path = os.path.join(args.source_dir, fn_frame[fn_frame['category'] == 'original'].iloc[0]['filepath'])
            mask_path = os.path.join(args.source_dir, fn_frame[fn_frame['category'] == 'mask'].iloc[0]['filepath'])

            job_kwargs_list.append(dict(
                video_path=video_path,
                mask_path=mask_path,
                img_size=args.img_size,
                wide_crop=args.wide_crop,
                output_dir=os.path.join(args.output_dir, filename[:filename.rfind('.')]),
            ))

        for _ in tqdm(pool.imap_unordered(job_proxy, job_kwargs_list), desc=f'Processing {split}', total=len(job_kwargs_list)):
            pass
            # T = len(video)
            #dset[start:(start + T)] = video
            # conf.append({'start': start, 'end': (start + T)})
            # start += T
        # conf = pandas.DataFrame(conf)
        # conf.to_json(os.path.join(args.output_dir, '{}.json'.format(split)), orient='records')


if __name__ == '__main__':
    main()