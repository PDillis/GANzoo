# From Cyril Diagne: http://cyrildiagne.com/

import os
import pickle
import argparse
import re

import numpy as np
import scipy

import PIL.Image
import moviepy.editor

import dnnlib
import dnnlib.tflib as tflib
import config


def load_Gs(model_path):
    with open(model_path, "rb") as file:
        _G, _D, Gs = pickle.load(file)
        # G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
    return _G, _D, Gs

def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_h, img_w, channels = images.shape

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros([grid_h * img_h, grid_w * img_w, channels], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y : y + img_h, x : x + img_w] = images[idx]
    return grid

def main(model_path, seed, size):

    tflib.init_tf()

    _, _, Gs = load_Gs(model_path=model_path)

    # Get the model name (for naming the video)
    model_name = re.search('([\w-]+).pkl', model_path).group(1)

    grid_size = [2, 2]
    image_shrink = 1
    image_zoom = 1
    duration_sec = 60.0
    smoothing_sec = 1.0
    mp4_fps = 20
    mp4_codec = 'libx264'
    mp4_bitrate = '5M'

    # Make the save dir:
    save_path = './{}/interpolation_videos/{}/'.format(config.result_dir, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    mp4_file = save_path + 'seed_{}-style_mixing_grid.mp4'.format(seed)
    minibatch_size = 8

    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(seed)

    # Generate latent vectors
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:] # [frame, image, channel, component]
    all_latents = random_state.randn(*shape).astype(np.float32)

    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7,
                              randomize_noise=False, output_transform=fmt)

        grid = create_image_grid(images, grid_size)
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2) # grayscale => RGB
        return grid

    # Generate video.
    video_clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    video_clip.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)




    # coarse
    duration_sec = 60.0
    smoothing_sec = 1.0
    mp4_fps = 20

    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_seed = 500
    random_state = np.random.RandomState(random_seed)


    w = size
    h = size
    #src_seeds = [601]
    dst_seeds = [700]
    style_ranges = ([0] * 7 + [range(8,16)]) * len(dst_seeds)

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    synthesis_kwargs = dict(output_transform=fmt, truncation_psi=0.7, minibatch_size=8)

    shape = [num_frames] + Gs.input_shape[1:] # [frame, image, channel, component]
    src_latents = random_state.randn(*shape).astype(np.float32)
    src_latents = scipy.ndimage.gaussian_filter(src_latents,
                                                smoothing_sec * mp4_fps,
                                                mode='wrap')
    src_latents /= np.sqrt(np.mean(np.square(src_latents)))

    dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)


    src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
    dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [seed, layer, component]
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)


    canvas = PIL.Image.new('RGB', (w * (len(dst_seeds) + 1), h * 2), 'white')

    for col, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), ((col + 1) * h, 0))

    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        src_image = src_images[frame_idx]
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), (0, h))

        for col, dst_image in enumerate(list(dst_images)):
            col_dlatents = np.stack([dst_dlatents[col]])
            col_dlatents[:, style_ranges[col]] = src_dlatents[frame_idx, style_ranges[col]]
            col_images = Gs.components.synthesis.run(col_dlatents, randomize_noise=False, **synthesis_kwargs)
            for row, image in enumerate(list(col_images)):
                canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * h, (row + 1) * w))
        return np.array(canvas)

    # Generate video.

    mp4_file = 'results/interpolate.mp4'
    mp4_codec = 'libx264'
    mp4_bitrate = '5M'

    video_clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    video_clip.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)


    w = size
    h = size

    # Different types of style ranges
    # Coarse styles: range(0, 4), Middle styles: range(4, 8), Fine: range(8, n)
    # where n will be defined, depending on size (max 18). This is because
    # there are 2 AdaIN blocks per layer (2 in 4x4, 2 in 8x8, ...)
    n = int(2 * (np.log2(size) - 1))
    # Depending on the input by the user:
    if generate_all:
        random = coarse = middle = fine = True
    if coarse:
        style_ranges = [range(0, 4)]
    if middle:
        style_ranges = [range(4, 8)]
    if fine:
        style_ranges = [range(8, n)]


    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    synthesis_kwargs = dict(output_transform=fmt, truncation_psi=0.7, minibatch_size=8)

    shape = [num_frames] + Gs.input_shape[1:] # [frame, image, channel, component]
    src_latents = random_state.randn(*shape).astype(np.float32)
    src_latents = scipy.ndimage.gaussian_filter(src_latents,
                                                smoothing_sec * mp4_fps,
                                                mode='wrap')
    src_latents /= np.sqrt(np.mean(np.square(src_latents)))

    dst_latents = np.stack([random_state.randn(Gs.input_shape[1])])


    src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
    dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [seed, layer, component]


    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        col_dlatents = np.stack([dst_dlatents[0]])
        col_dlatents[:, style_ranges[0]] = src_dlatents[frame_idx, style_ranges[0]]
        col_images = Gs.components.synthesis.run(col_dlatents, randomize_noise=False, **synthesis_kwargs)
        return col_images[0]

def parse():
    parser = argparse.ArgumentParser(
        description="Circular interpolation with respect to a point."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to trained model.",
        required=True
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed.",
        default=1000
    )
    parser.add_argument(
        "--size",
        type=int,
        help="Size of generated images.",
        default=512
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    main(model_path=args.model_path, seed=args.seed, size=args.size)
