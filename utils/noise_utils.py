import torch
import numpy as np
from tqdm import tqdm


# helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
def denoise_add_noise(x, t, pred_noise, a_t, ab_t, b_t, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise


# sample using standard algorithm
@torch.no_grad()
def sample_ddpm(nn_model, n_sample, height, timesteps, a_t, ab_t, b_t, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, height).cuda()

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].cuda()

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t)  # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, a_t, ab_t, b_t, z)
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate


# incorrectly sample without adding in noise
@torch.no_grad()
def sample_ddpm_incorrect(nn_model, n_sample, height, timesteps, a_t, ab_t, b_t):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, height).cuda()

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in tqdm(range(timesteps, 0, -1), desc='sampling timestep'):

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].cuda()

        # don't add back in noise
        z = 0

        eps = nn_model(samples, t)  # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, a_t, ab_t, b_t, z)
        if i % 20 == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate
