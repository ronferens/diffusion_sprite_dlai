import torch
import matplotlib.pyplot as plt
from models.ContextUnet import ContextUnet
from utils.noise_utils import sample_ddpm, sample_ddpm_incorrect
from utils.plot_utils import plot_samples, animate_sampling

# ############################
# Set up
# ############################
# diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

# network hyperparameters
n_feat = 64  # 64 hidden dimension feature
n_cfeat = 5  # context vector is of size 5
height = 16  # 16x16 image
save_dir = './models/weights/'
n_samples = 32
n_disp_rows = 4

# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1).cuda() + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

# construct model
nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height)

# load in model weights and set to eval mode
nn_model.load_state_dict(torch.load(f"{save_dir}/model_trained.pth"))
nn_model.cuda()
nn_model.eval()
print("Loaded in Model")

# Demonstrate incorrectly sample without adding the 'extra noise'
# ---------------------------------------------------------------
samples, intermediate = sample_ddpm_incorrect(nn_model, n_samples, height, timesteps, a_t, ab_t, b_t)
plot_samples(samples.cpu().numpy(), n_disp_rows, title='DDPM - Incorrectly Sample without Adding in Noise',
             save_path='./outputs', prefix='ddpm_incorrect')
animate_sampling(intermediate, n_disp_rows, title='DDPM - Incorrectly Sample without Adding in Noise',
                 save_path='./outputs', prefix='ddpm_incorrect', fps=30)

# Demonstrate correctly sample with 'extra noise'
# -----------------------------------------------
samples, intermediate = sample_ddpm(nn_model, n_samples, height, timesteps, a_t, ab_t, b_t)
plot_samples(samples.cpu().numpy(), n_disp_rows, title='DDPM - Correct Samples', save_path='./outputs',
             prefix='ddpm_correct')
animate_sampling(intermediate, n_disp_rows, title='DDPM - Correct Samples', save_path='./outputs',
                 prefix='ddpm_correct', fps=30)
