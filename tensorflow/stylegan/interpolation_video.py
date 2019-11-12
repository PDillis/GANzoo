# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""
# Import libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pickle

import numpy as np
import scipy

import dnnlib
import dnnlib.tflib as tflib
import config

import re
import os
import sys
import glob
import argparse

import PIL.Image


def load_Gs(path):
    with open(path, "rb") as file:
        _G, _D, Gs = pickle.load(file)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
    return Gs


def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(
        list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype
    )
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid


def generate_interpolation_video(
    model_path,
    desc,
    cols,
    rows,
    image_shrink=1,
    image_zoom=1,
    duration_sec=30.0,
    smoothing_sec=3.0,
    mp4_fps=30,
    mp4_codec="libx265",
    mp4_bitrate="16M",
    seed=1000,
    minibatch_size=8,
):
    # Let's get the model name (without the .pkl):
    model_name = re.search('([\w-]+).pkl', model_path).group(1)
    # By default, the video will be saved in the ./stylegan/results/ subfolder
    save_path = "./{}/interpolation_videos/{}/".format(config.result_dir, model_name)
    # If save_path doesn't exist, create it:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the video as ./results/interpolation_videos/slerp/model_name/seed-#-slerp.mp4:
    mp4 = save_path + "seed_{}-slerp.mp4".format(seed)
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(seed)

    print('Loading network from "%s"...' % model_path)
    Gs = load_Gs(path=model_path)

    print("Generating latent vectors...")
    grid_size = [cols, rows]
    # [frame, image, channel, component]:
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:]
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(
        all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode="wrap"
    )
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        # Get the images (with labels = None)
        images = Gs.run(
            latents,
            None,
            minibatch_size=minibatch_size,
            num_gpus=1,
            out_mul=127.5,
            out_add=127.5,
            out_shrink=image_shrink,
            out_dtype=np.uint8,
            truncation_psi=0.7,
            randomize_noise=False
        )
        grid = create_image_grid(images, grid_size).transpose(1, 2, 0)  # HWC
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)  # grayscale => RGB
        return grid

    # Generate video.
    import moviepy.editor  # pip install moviepy

    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    videoclip.write_videofile(mp4, fps=mp4_fps, codec="libx264", bitrate=mp4_bitrate)


def main(model_path, seed, duration_sec, cols, rows, fps):
    # Initialize TensorFlow.
    tflib.init_tf()

    generate_interpolation_video(
        model_path=model_path,
        desc="interpolation_video",
        cols=cols,
        rows=rows,
        duration_sec=duration_sec,
        seed=seed,
        mp4_fps=fps,
    )


def parse():
    parser = argparse.ArgumentParser(
        description="Interpolation of latent space in a trained StyleGAN model."
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for the generated images",
        default=42
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pretrained model.",
        required=True
    )
    parser.add_argument(
        "--duration_sec",
        type=float,
        help="Duration in seconds of the video.",
        default=60.0,
    )
    parser.add_argument(
        "--cols",
        type=int,
        help="Number of columns in generated video.",
        default=3
    )
    parser.add_argument(
        "--rows",
        type=int,
        help="Number of rows in generated video.",
        default=2
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="FPS of video (use lower values for unstable latent space and prevent headaches).",
        default=30
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    # Initialize TensorFlow
    tflib.init_tf()

    main(
        model_path=args.model_path,
        seed=args.seed,
        duration_sec=args.duration_sec,
        cols=args.cols,
        rows=args.rows,
        fps=args.fps,
    )
