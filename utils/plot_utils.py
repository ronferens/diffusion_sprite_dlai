import numpy as np
import matplotlib.pyplot as plt
import tempfile
import torch
from os.path import join
import cv2
from os import listdir
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import ImageGrid


def norm_torch(x_all):
    # runs unity norm on all timesteps of all samples
    # input is (n_samples, 3,h,w), the torch image format
    x = x_all.cpu().numpy()
    xmax = x.max((2, 3))
    xmin = x.min((2, 3))
    xmax = np.expand_dims(xmax, (2, 3))
    xmin = np.expand_dims(xmin, (2, 3))
    nstore = (x - xmin) / (xmax - xmin)
    return torch.from_numpy(nstore)


def unorm(x):
    # unity norm. results in range of [0,1]
    # assume x (h,w,3)
    xmax = x.max((0, 1))
    xmin = x.min((0, 1))
    return (x - xmin) / (xmax - xmin)


def norm_all_timestamp(store):
    # runs unity norm on all samples at a given timestamp
    nstore = np.zeros_like(store)
    for idx in range(store.shape[0]):
        nstore[idx] = unorm(store[idx])
    return nstore


def norm_all(store, n_t, n_s):
    # runs unity norm on all timesteps of all samples
    nstore = np.zeros_like(store)
    for t in range(n_t):
        for s in range(n_s):
            nstore[t, s] = unorm(store[t, s])
    return nstore


def create_samples_grid(samples, nrows, fig=None, grid=None):
    ncols = samples.shape[0] // nrows

    # Change to Numpy image format (h,w,channels) vs (channels,h,w)
    samples = np.transpose(samples, [0, 2, 3, 1])

    # Normalize the image between [0, 1]
    norm_samples = norm_all_timestamp(samples)

    if fig is None or grid is None:
        fig = plt.figure(figsize=(ncols, nrows))

        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )

    for ax, im in zip(grid, norm_samples):
        # Iterating over the grid returns the Axes.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(im)

    return fig, grid


def plot_samples(samples, nrows, title=None, save_path=None, prefix=None):
    # Plotting all samples
    fig, grid = create_samples_grid(samples, nrows)

    # Adding title if given
    if title is not None:
        fig.suptitle(title)

    # Saving the plot if a save_path was specified
    if save_path is not None:
        plt.savefig(join(save_path, 'samples.png' if prefix is None else f'{prefix}_samples.png'))

    plt.show()
    plt.close()


def animate_sampling(intermediate_imgs, nrows, title=None, save_path=None, prefix=None, fps=30):
    fig = None
    grid = None
    image_list = []

    with tempfile.TemporaryDirectory() as tmpdirname:
        for t in tqdm(range(intermediate_imgs.shape[0]), desc='generating animation frames'):
            fig, grid = create_samples_grid(intermediate_imgs[t], nrows, fig, grid)
            fig.suptitle(title + f' Index={t}' if title is not None else f'Index={t}')
            tmp_filename = f'{t}.png'
            fig.savefig(join(tmpdirname, tmp_filename))
            image_list.append(tmp_filename)
        plt.close(fig)

        if len(image_list) == 0:
            raise ValueError("No images found in the specified folder.")

        # Get the first image to determine dimensions
        first_image = cv2.imread(join(tmpdirname, image_list[0]))
        height, width, _ = first_image.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can also use other codecs like 'MJPG', 'H264', etc.
        video_name = join(save_path, 'anim_sampling.avi' if prefix is None else f'{prefix}_anim_sampling.avi')
        out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        for image_file in image_list:
            image_path = join(tmpdirname, image_file)
            frame = cv2.imread(image_path)

            if frame is None:
                continue

            out.write(frame)

        # Release the VideoWriter and destroy any OpenCV windows
        out.release()
        cv2.destroyAllWindows()
