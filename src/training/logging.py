import os
from typing import List, Callable, Optional, Dict
from multiprocessing.pool import ThreadPool

from PIL import Image
import torch
from torch import Tensor
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import utils
import torchvision.transforms.functional as TVF

#----------------------------------------------------------------------------

@torch.no_grad()
def generate_videos(
    G: Callable, z: Tensor, c: Tensor, ts: Tensor, motion_z: Optional[Tensor]=None,
    noise_mode='const', truncation_psi=1.0, verbose: bool=False, as_grids: bool=False, batch_size_num_frames: int=100) -> Tensor:

    assert len(ts) == len(z) == len(c), f"Wrong shape: {ts.shape}, {z.shape}, {c.shape}"
    assert ts.ndim == 2, f"Wrong shape: {ts.shape}"

    G.eval()
    videos = []

    if c.shape[1] > 0 and truncation_psi < 1:
        num_ws_to_average = 1000
        c_for_avg = c.repeat_interleave(num_ws_to_average, dim=0) # [num_classes * num_ws_to_average, num_classes]
        z_for_avg = torch.randn(c_for_avg.shape[0], G.z_dim, device=z.device) # [num_classes * num_ws_to_average, z_dim]
        w = G.mapping(z_for_avg, c=c_for_avg)[:, 0] # [num_classes * num_ws_to_average, w_dim]
        w_avg = w.view(-1, num_ws_to_average, G.w_dim).mean(dim=1) # [num_classes, w_dim]

    iters = range(len(z))
    iters = tqdm(iters, desc='Generating videos') if verbose else iters

    if motion_z is None and not G.synthesis.motion_encoder is None:
        motion_z = G.synthesis.motion_encoder(c=c, t=ts)['motion_z'] # [...any...]

    for video_idx in iters:
        curr_video = []


        for curr_ts in ts[[video_idx]].split(batch_size_num_frames, dim=1):
            curr_z = z[[video_idx]] # [1, z_dim]
            curr_c = c[[video_idx]] # [1, c_dim]
            curr_motion_z = motion_z[[video_idx]]

            if curr_c.shape[1] > 0 and truncation_psi < 1:
                curr_w = G.mapping(curr_z, c=curr_c, truncation_psi=1) # [1, num_ws, w_dim]
                curr_w = truncation_psi * curr_w + (1 - truncation_psi) * w_avg.unsqueeze(1) # [1, num_ws, w_dim]
                out = G.synthesis(
                    ws=curr_w,
                    c=curr_c,
                    t=curr_ts,
                    motion_z=curr_motion_z,
                    noise_mode=noise_mode) # [1 * curr_num_frames, 3, h, w]
            else:
                out = G(
                    z=curr_z,
                    c=curr_c,
                    t=curr_ts,
                    motion_z=curr_motion_z,
                    truncation_psi=truncation_psi,
                    noise_mode=noise_mode) # [1 * curr_num_frames, 3, h, w]

            out = (out * 0.5 + 0.5).clamp(0, 1).cpu() # [1 * curr_num_frames, 3, h, w]
            curr_video.append(out)

        videos.append(torch.cat(curr_video, dim=0))

    videos = torch.stack(videos) # [len(z), video_len, c, h, w]

    if as_grids:
        frame_grids = videos.permute(1, 0, 2, 3, 4) # [video_len, len(z), c, h, w]
        frame_grids = [utils.make_grid(fs, nrow=int(np.sqrt(len(z)))) for fs in frame_grids] # [video_len, 3, grid_h, grid_w]

        return torch.stack(frame_grids)
    else:
        return videos

#----------------------------------------------------------------------------

def run_batchwise(fn: Callable, data_kwargs: Dict[str, Tensor], batch_size: int, **kwargs) -> Tensor:
    data_kwargs = {k: v for k, v in data_kwargs.items() if not v is None}
    seq_len = len(data_kwargs[list(data_kwargs.keys())[0]])
    result = []

    for i in range((seq_len + batch_size - 1) // batch_size):
        curr_data_kwargs = {k: d[i * batch_size: (i+1) * batch_size] for k, d in data_kwargs.items()}
        result.append(fn(**curr_data_kwargs, **kwargs))

    return torch.cat(result, dim=0)

#----------------------------------------------------------------------------

def save_video_frames_as_mp4(frames: List[Tensor], fps: int, save_path: os.PathLike, verbose: bool=False):
    # Load data
    frame_h, frame_w = frames[0].shape[1:]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (frame_w, frame_h))
    frames = tqdm(frames, desc='Saving videos') if verbose else frames
    for frame in frames:
        assert frame.shape[0] == 3, "RGBA/grayscale images are not supported"
        frame = np.array(TVF.to_pil_image(frame))
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Uncomment this line to release the memory.
    # It didn't work for me on centos and complained about installing additional libraries (which requires root access)
    # cv2.destroyAllWindows()
    video.release()

#----------------------------------------------------------------------------

def save_video_frames_as_frames(frames: List[Tensor], save_dir: os.PathLike, time_offset: int=0):
    os.makedirs(save_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        save_path = os.path.join(save_dir, f'{i + time_offset:06d}.jpg')
        TVF.to_pil_image(frame).save(save_path, q=95)

#----------------------------------------------------------------------------

def save_video_frames_as_frames_parallel(frames: List[np.ndarray], save_dir: os.PathLike, time_offset: int=0, num_processes: int=1):
    assert num_processes > 1, "Use `save_video_frames_as_frames` if you do not plan to use num_processes > 1."
    os.makedirs(save_dir, exist_ok=True)
    # We are fine with the ThreadPool instead of Pool since most of the work is I/O
    pool = ThreadPool(processes=num_processes)
    save_paths = [os.path.join(save_dir, f'{i + time_offset:06d}.jpg') for i in range(len(frames))]
    pool.map(save_jpg_mp_proxy, [(f, p) for f, p in zip(frames, save_paths)])

#----------------------------------------------------------------------------

def save_jpg_mp_proxy(args):
    return save_jpg(*args)

#----------------------------------------------------------------------------

def save_jpg(x: np.ndarray, save_path: os.PathLike):
    Image.fromarray(x).save(save_path, q=95)

#----------------------------------------------------------------------------
