import tensorflow as tf
import numpy as np
import pickle

def load_pretrained(path):
    with open(path, "rb") as file:
        G, D, Gs = pickle.load(file)
    return G, D, Gs

def generate_interpolation_video(path, grid_size=[5,4], image_shrink=1, image_zoom=1, duration_sec=30.0, smoothing_sec=1.0, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M', random_seed=1000, minibatch_size=8):
    network_pkl = path

    mp4 = './a2d2_256-lerp.mp4'
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = load_pretrained(path)

    print('Generating latent vectors...')
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:] # [frame, image, channel, component]
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        grid = misc.create_image_grid(images, grid_size).transpose(1, 2, 0) # HWC
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2) # grayscale => RGB
        return grid

    # Generate video.
    import moviepy.editor # pip install moviepy
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(mp4, fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)

if __name__=='__main__':
    a2d2_256_pkl = './network-snapshot-008566.pkl'
    generate_interpolation_video(a2d2_256_pkl)
