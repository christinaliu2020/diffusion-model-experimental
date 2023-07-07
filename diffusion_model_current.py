# -*- coding: utf-8 -*-
"""Diffusion_model_current.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CsV5EDxlOkjerVjPQkZ1LZzcCDoAsStt
"""



#@title Defining a time-dependent score-based model (double click to expand or collapse)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision


class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std

  def forward(self, x, t):
    # Obtain the Gaussian random feature embedding for t
    embed = self.act(self.embed(t))
    # Encoding path
    h1 = self.conv1(x)
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)

    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    h = h / self.marginal_prob_std(x, t, 0)[1][:, None, None, None]
    return h

"""## Training with Weighted Sum of Denoising Score Matching Objectives

Now let's get our hands dirty on training. First of all, we need to specify an SDE that perturbs the data distribution $p_0$ to a prior distribution $p_T$. We choose the following SDE
\begin{align*}
d \mathbf{x} = \sigma^t d\mathbf{w}, \quad t\in[0,1]
\end{align*}
In this case,
\begin{align*}
p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0)) = \mathcal{N}\bigg(\mathbf{x}(t); \mathbf{x}(0), \frac{1}{2\log \sigma}(\sigma^{2t} - 1) \mathbf{I}\bigg)
\end{align*}
and we can choose the weighting function $\lambda(t) = \frac{1}{2 \log \sigma}(\sigma^{2t} - 1)$.

When $\sigma$ is large, the prior distribution, $p_{t=1}$ is
\begin{align*}
\int p_0(\mathbf{y})\mathcal{N}\bigg(\mathbf{x}; \mathbf{y}, \frac{1}{2 \log \sigma}(\sigma^2 - 1)\mathbf{I}\bigg) d \mathbf{y} \approx \mathbf{N}\bigg(\mathbf{x}; \mathbf{0}, \frac{1}{2 \log \sigma}(\sigma^2 - 1)\mathbf{I}\bigg),
\end{align*}
which is approximately independent of the data distribution and is easy to sample from.

Intuitively, this SDE captures a continuum of Gaussian perturbations with variance function $\frac{1}{2 \log \sigma}(\sigma^{2t} - 1)$. This continuum of perturbations allows us to gradually transfer samples from a data distribution $p_0$ to a simple Gaussian distribution $p_1$.
"""

import abc
import functools
import scipy.optimize as opt
#@title Set up the SDE

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self):
    super().__init__()

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    """Drift f(x, t) and diffusion g(t)."""
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Mean and std. dev. of the marginal probability of x(t)|x(eps).

    In original paper by Yang Song, eps = 0.

    Args:
      x: Mini-batch of data, an array of shape (b, w, h, c).
      t: Mini-batch of time values, an array of shape (b,).
      eps: Lowest diffusion time, a scalar. In original implementation
        by Yang Song, eps = 0.
    """
    pass


class VPSDE(SDE):
  """Variance Preserving SDE."""

  def __init__(self, beta_min=0.1, beta_max=20):
    super().__init__()
    self.beta_0 = beta_min
    self.beta_1 = beta_max

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t, eps):
    log_mean_coeff = (
      -0.25 * (t ** 2 - eps ** 2) * (self.beta_1 - self.beta_0)
      - 0.5 * (t - eps) * self.beta_0)
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def solve_t_given_std(self, x, std, eps):

    def equation(t):
        log_mean_coeff = (
            -0.25 * (t ** 2 - eps ** 2) * (self.beta_1 - self.beta_0)
            - 0.5 * (t - eps) * self.beta_0
        )
        return 0.5 * np.log(1. - std ** 2) - log_mean_coeff

    t_solution = opt.fsolve(equation, eps)  # Use numerical solver from scipy.optimize
    return t_solution.item()


class subVPSDE(SDE):
  """sub-VP SDE."""

  def __init__(self, beta_min=0.1, beta_max=20):
    super().__init__()
    self.beta_0 = beta_min
    self.beta_1 = beta_max

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t, eps):
    log_mean_coeff = (
      -0.25 * (t ** 2 - eps ** 2) * (self.beta_1 - self.beta_0)
      - 0.5 * (t - eps) * self.beta_0)
    mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std





class VESDE(SDE):
  """Variance Exploding SDE."""

  def __init__(self, sigma_min=0.01, sigma_max=50):
    super().__init__()
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t, eps):
    sigma2_t = self.sigma_min ** 2 * (self.sigma_max / self.sigma_min) ** (2 * t)
    sigma2_eps = self.sigma_min ** 2 * (self.sigma_max / self.sigma_min) ** (2 * eps)
    std = torch.sqrt(sigma2_t - sigma2_eps)
    mean = x
    return mean, std
eps = 0.01
sde = VPSDE()
marginal_prob_fn = sde.marginal_prob
sde_fn = sde.sde
solve_t_given_std_fn = sde.solve_t_given_std
marginal_prob_std = functools.partial(marginal_prob_fn)
diffusion_coeff_fn = functools.partial(sde_fn)
solve_t_given_std = functools.partial(solve_t_given_std_fn)

#@title Define the loss function (double click to expand or collapse)
#The original loss function assumes eps = 0, so to account for eps > 0, we need to change the loss function:

def loss_fn(model, x, marginal_prob, eps):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the mean and standard deviation of
      the perturbation kernel.
    eps: Smallest diffusion time.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  mean, std = marginal_prob(x, random_t, eps)
  #(std)
  perturbed_x = mean + z * std[:, None, None, None]

  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss, std

#@title Define the Euler-Maruyama sampler (double click to expand or collapse)

import matplotlib.pyplot as plt
import math
## The number of sampling steps.
num_steps =  500#@param {'type':'integer'}
def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           solve_t_given_std,
                           diffusion_coeff,
                           batch_size=64,
                           num_steps=num_steps,
                           device='cuda',
                           eps_val = 0.0001,
                           t_val=0.0001):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
    #CHANGED TO 0.0001 FROM 0.001

  Returns:
    Samples.
  """
  t = torch.ones(batch_size, device=device)
  y = torch.randn(batch_size, 1, 28, 28, device=device)
  y = y.to(device)

  init_x = torch.randn(batch_size, 1, 28, 28, device=device) #\
    #* marginal_prob_std(y, t, 0)[1][:, None, None, None]

  #t_val = solve_t_given_std(y, train_noise, 0) #train_noise is std dev
  print("t_val:", t_val)
  print("eps_val:", eps_val)
  time_steps = torch.linspace(1., 0.0001, num_steps, device=device)

  #print("timestep list", time_steps)

  step_size = time_steps[0] - time_steps[1]
  x = init_x
  x = x.to(device)
  a = (0.0001-1)/(num_steps-1)
  b = (eps_val-1)//a # sampled to eps value
  c = (t_val-1)//a #sampled to trained noise level
  d = (0.001-1)//a #sampled to 0.001
  e = (0.01-1)//a #sampled to 0.01
  i=0
  #eps_sample_x = torch.empty((64, 1, 28,28), dtype=torch.int64)
  with torch.no_grad():
    for time_step in tqdm.notebook.tqdm(time_steps):
      if i == b:
        #plt.imshow(x[0][0].cpu())
        print(f"stop at eps {eps_val}",time_step)
        eps_sample_x = x #store image stopped at eps value
      if i == c:
        noise_sample_x = x
        print("stop at trained noise",time_step)
      if i == e:
        sample_one_x = x
        print("stop at 0.01")
      if i == d:
        sample_two_x = x
        print("stop at 0.001")
      i+=1
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(x, batch_time_step)[1]
      mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size

      x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

      #print("mean x:", mean_x)
  # Do not include any noise in the last sampling step.
  #print("final returned mean_x", mean_x)
  return noise_sample_x, eps_sample_x, mean_x, sample_one_x, sample_two_x



"""Sample images to trained value and epsilon value
Half_sampled_img : img sampled to

PSNR noise computing function, larger value/sigma -> more noise
"""

import cv2
import math
from scipy.signal import convolve2d

def estimate_noise(I):

  H, W = I.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return sigma


import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
import numpy as np
import random
import functools
import matplotlib.pyplot as plt

def train_model(eps, t_val):

  #@title Training (double click to expand or collapse)
  #eps = 6.0214e-02

  #@title Set up the SDE



  # noise_values = [0.01, 0.04, 0.05, 0.06, 0.07, 0.08, 0.6]
  # eps_values = [0.04, 0.05, 0.06, 0.07, 0.08, 0.2, 0.6]

  noise_values = eps_list
  eps_values = eps_list




  device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

  score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std = marginal_prob_std))
  score_model = score_model.to(device)

  n_epochs =   50#@param {'type':'integer'}
  ## size of a mini-batch
  batch_size =  64 #@param {'type':'integer'}
  ## learning rate
  lr=1e-4 #@param {'type':'number'}

  dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4) #possibly chnage to 2

  losslist = []

  epoch_list = []

  # Get noise std. dev. at diffusion time `eps`.
  std_dev = sde.marginal_prob(torch.ones(1, 1),torch.ones(1) * t_val,0)[1]
  std_dev = std_dev[0]
  std_dev = std_dev.to(device)
  optimizer = Adam(score_model.parameters(), lr=lr)
  tqdm_epoch = tqdm.tnrange(n_epochs)
  counter = 0

  for epoch in tqdm_epoch:
    epoch_list.append(epoch)
    avg_loss = 0.
    num_items = 0.

    for x, y in data_loader:
      x = x.to(device)
      if(counter == 0):
        raw_img = x[0][0].cpu()

      counter = counter + 1;
      # Add noise.
      #noise = torch.randn_like(x)
      # std_dev = sde.marginal_prob(x, torch.ones(1)*t_val, 0)[1]
      # std_dev = std_dev.to(device)
      noise = std_dev * torch.rand_like(x)
      noisy_x = x + noise
      #noise = noise.to(device)
      #noisy_x = x + std_eps * noise

      loss, std = loss_fn(score_model, noisy_x, marginal_prob_std, eps)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      avg_loss += loss.item() * x.shape[0]
      num_items += x.shape[0]


    # Print the averaged training loss so far.
    losslist.append(avg_loss/num_items)
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Update the checkpoint after each epoch of training.
    torch.save(score_model.state_dict(), f'ckpt.pth')
    t = torch.ones(batch_size, device=device)

  half_noised_img = noisy_x[0][0].cpu()


  fig, axs = plt.subplots(1, 2, figsize=(10, 8))
  axs[0].axis('off')
  axs[1].axis('off')


  img1 = axs[0].imshow(half_noised_img, vmin=0, vmax=1) #slightly noised image
  extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  plt.savefig(f'/home/christina/Downloads/images/slightly noised image, noise: {noise_values.index(t_val)}, eps: {eps_values.index(eps)}', bbox_inches=extent)
  axs[0].set_title(f'slightly noised image t_val: {t_val}, eps: {eps}')

  img2 = axs[1].imshow(raw_img, vmin=0, vmax=1) #raw image
  extent2 = axs[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  plt.savefig(f'/home/christina/Downloads/images/raw image', bbox_inches=extent2)
  axs[1].set_title(f'raw image')

  cbar = fig.colorbar(img2, ax=axs)

  plt.savefig(f'/home/christina/Downloads/images/Trained Figure noise idx {noise_values.index(t_val)}, eps idx {eps_values.index(eps)}')
  #plt.show()



  return score_model#, std_list, epoch_list

"""Epsilon loop"""

# Commented out IPython magic to ensure Python compatibility.
from PIL import Image
import numpy

# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

#epsilon_values= [0.00625, 0.0075, 0.00875, 0.01, 0.01125, 0.0125, 0.01375, 0.01]#, 0.015, 0.01625, 0.0175]
# eps_values = [0.04, 0.05, 0.06, 0.07, 0.08, 0.2, 0.6]
# noise_values = [0.01, 0.04, 0.05, 0.06, 0.07, 0.08, 0.6]
# #eps = 6.0214e-02
#
# used for testing table of t and eps values
# values = [0.04, 0.05, 0.06, 0.07, 0.08]
# list = []
# for i in range(0, 5):
#   for j in range(0, 5):
#     val = values[i], values[j]
#     list.append(val)
# val2 = 0.06, 0.01
# val3 = 0.2, 0.01
# list.append(val2)
# list.append(val3)
# val4 = 0.6, 0.6
# list.append(val4)

lowest_eps = 0.001
highest_eps = 0.5
n_eps = 20
log_eps = np.random.uniform(np.log10(lowest_eps), np.log10(highest_eps),size=n_eps)
eps_list = 10**log_eps
eps_list = eps_list.tolist()
list =[]
for i in range(20):
  val = eps_list[i], eps_list[i]
  list.append(val)

print("list of eps and t values used: ", list)
noise_values = eps_list
eps_values = eps_list
#noise_values stores std dev of noise levels added to perturbe image in training
#rows, cols = (4, 8)
#noise_array = [[0]*cols]*rows
noise_list = [] #noise of imgs sampled to trained noise level
noise_list2 = [] #noise of imgs sampled to eps value, trained on diff noise levels
noise_list3 = [] #noise of imgs sampled to t-value 0.0001, for different trained noise levels
for eps, t_val in list:
  score_model = train_model(eps, t_val)

  raw_img = cv2.imread('/home/christina/Downloads/images/raw image.png')
  raw_img_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
  raw_noise = estimate_noise(raw_img_gray)
  print("raw image noise:", raw_noise)

  half_noise_img = cv2.imread(f'/home/christina/Downloads/images/slightly noised image, noise: {noise_values.index(t_val)}, eps: {eps_values.index(eps)}.png')
  half_noise_img_gray = cv2.cvtColor(half_noise_img, cv2.COLOR_BGR2GRAY)
  half_noise = estimate_noise(half_noise_img_gray)
  print(f"slightly noised image trained on noise t-val {t_val}, eps {eps} Noise:", half_noise)

  #Start of sampling
  ## Load the pre-trained checkpoint from disk.
  device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
  ckpt = torch.load('ckpt.pth', map_location=device)
  score_model.load_state_dict(ckpt)

  sample_batch_size = 64 #@param {'type':'integer'}
  sampler = Euler_Maruyama_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}
#   %matplotlib inline
  import matplotlib.pyplot as plt
  ## Generate samples using the specified sampler.
  samples = sampler(score_model,
                    marginal_prob_std,
                    solve_t_given_std,
                    diffusion_coeff_fn,
                    sample_batch_size,
                    device=device,
                    eps_val= eps,
                    t_val = t_val)


  fig, axs = plt.subplots(2, 5, figsize=(25, 8))
  # axs[0].axis('off')
  # axs[1].axis('off')
  # axs[2].axis('off')
  # axs[3].axis('off')
  # axs[4].axis('off')

  ax1 = axs[0, 0]  # First subplot in the first row
  ax2 = axs[0, 1]  # Second subplot in the first row
  ax3 = axs[0, 2]  # Third subplot in the first row
  ax4 = axs[0, 3]  # Fourth subplot in the first row
  ax5 = axs[0, 4]  # Fifth subplot in the first row
  ax6 = axs[1, 0]  # First subplot in the second row
  ax7 = axs[1, 1]  # Second subplot in the second row
  ax8 = axs[1, 2]  # Third subplot in the second row
  ax9 = axs[1, 3]  # Fourth subplot in the second row
  ax10 = axs[1, 4]

  for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]:
    ax.axis('off')

  img2 = ax1.imshow(samples[1][0][0].cpu(), vmin=0, vmax=1) #sampled to eps
  extent2 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  plt.savefig(f'/home/christina/Downloads/images/Sampled 1 to eps, trained noise t idx:{noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}', bbox_inches=extent2)
  ax1.set_title(f'Sampled to eps: {eps}')

  img1 = ax2.imshow(samples[0][0][0].cpu(), vmin=0, vmax=1) #sampled to t-val of noise std_dev
  extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  plt.savefig(f'/home/christina/Downloads/images/Sampled 1 to noise t, trained noise t idx: {noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}', bbox_inches=extent)
  ax2.set_title(f'Sampled to noise t: {t_val}')

  img3 = ax3.imshow(samples[3][0][0].cpu(), vmin=0, vmax=1)  # sampled to 0.01
  extent3 = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  plt.savefig(
    f'/home/christina/Downloads/images/Sampled 1 to point zero one, trained noise t idx: {noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}',
    bbox_inches=extent3)
  ax3.set_title(f'Sampled to 0.01')

  img4 = ax4.imshow(samples[4][0][0].cpu(), vmin=0, vmax=1)  # sampled to 0.001
  extent4 = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  plt.savefig(
    f'/home/christina/Downloads/images/Sampled 1 to point zero zero one, trained noise t idx: {noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}',
    bbox_inches=extent4)
  ax4.set_title(f'Sampled to 0.001')

  img5 = ax5.imshow(samples[2][0][0].cpu(), vmin=0, vmax=1) #fully sampled to 0.0001
  extent5 = ax5.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  plt.savefig(f'/home/christina/Downloads/images/Sampled 1 to point zero zero zero one, trained noise t idx:{noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}', bbox_inches=extent5)
  ax5.set_title(f'Sampled to 0.0001')

  cbar = fig.colorbar(img5, ax=axs)



  #second row
  img6 = ax6.imshow(samples[1][1][0].cpu(), vmin=0, vmax=1)  # sampled to eps
  extent6 = ax6.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  plt.savefig(
    f'/home/christina/Downloads/images/Sampled 2 to eps, trained noise t idx:{noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}',
    bbox_inches=extent6)
  ax6.set_title(f'Sampled to eps: {eps}')

  img7 = ax7.imshow(samples[0][1][0].cpu())#, vmin=0, vmax=1)  # sampled to t-val of noise std_dev
  extent7 = ax7.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  plt.savefig(
    f'/home/christina/Downloads/images/Sampled 2 to noise t, trained noise t idx: {noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}',
    bbox_inches=extent7)
  ax7.set_title(f'Sampled to noise t: {t_val}')

  img8 = ax8.imshow(samples[3][1][0].cpu())#, vmin=0, vmax=1)  # sampled to 0.01
  extent8 = ax8.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  plt.savefig(
    f'/home/christina/Downloads/images/Sampled 2 to point zero one, trained noise t idx: {noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}',
    bbox_inches=extent8)
  ax8.set_title(f'Sampled to 0.01')

  img9 = ax9.imshow(samples[4][1][0].cpu())#, vmin=0, vmax=1)  # sampled to 0.001
  extent9 = ax9.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  plt.savefig(
    f'/home/christina/Downloads/images/Sampled 2 to point zero zero one, trained noise t idx: {noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}',
    bbox_inches=extent9)
  ax9.set_title(f'Sampled to 0.001')

  img10 = ax10.imshow(samples[2][1][0].cpu())#, vmin=0, vmax=1)  # fully sampled to 0.0001
  extent10 = ax10.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  plt.savefig(
    f'/home/christina/Downloads/images/Sampled 2 to point zero zero zero one, trained noise t idx:{noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}',
    bbox_inches=extent10)
  ax10.set_title(f'Sampled to 0.0001')


  fig.suptitle(f'Trained on noise t {t_val} and eps {eps}', fontsize=16)

  plt.savefig(f'/home/christina/Downloads/images/Sampled Figures noise t idx {noise_values.index(t_val)}, eps idx {eps_values.index(eps)}')
  #plt.show()


  #sample 1 noise
  print("sample 1 noise estimates:")
  img2 = cv2.imread(f'/home/christina/Downloads/images/Sampled 1 to eps, trained noise t idx:{noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}.png')
  img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  noise2 = estimate_noise(img_gray2)
  print(f'Sampled 1 to eps {eps}, trained noise t:{t_val}, eps {eps} Noise estimate:', noise2)


  img1_noise = cv2.imread(f'/home/christina/Downloads/images/Sampled 1 to noise t, trained noise t idx: {noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}.png')
  img_gray = cv2.cvtColor(img1_noise, cv2.COLOR_BGR2GRAY)
  noise1 = estimate_noise(img_gray)
  print(f'Sampled 1 to noise t {t_val}, trained noise t:{t_val}, eps {eps} Noise estimate:', noise1)

  img3_noise = cv2.imread(
    f'/home/christina/Downloads/images/Sampled 1 to point zero one, trained noise t idx: {noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}.png')
  img_gray3 = cv2.cvtColor(img3_noise, cv2.COLOR_BGR2GRAY)
  noise3 = estimate_noise(img_gray3)
  print(f'Sampled 1 to 0.01, trained noise t:{t_val}, eps {eps} Noise estimate:', noise3)

  img4_noise = cv2.imread(
    f'/home/christina/Downloads/images/Sampled 1 to point zero zero one, trained noise t idx: {noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}.png')
  img_gray4 = cv2.cvtColor(img4_noise, cv2.COLOR_BGR2GRAY)
  noise4 = estimate_noise(img_gray4)
  print(f'Sampled 1 to 0.001, trained noise t:{t_val}, eps {eps} Noise estimate:', noise4)


  img5_noise = cv2.imread(f'/home/christina/Downloads/images/Sampled 1 to point zero zero zero one, trained noise t idx:{noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}.png')
  img_gray5 = cv2.cvtColor(img5_noise, cv2.COLOR_BGR2GRAY)
  noise5 = estimate_noise(img_gray5)
  print(f'Sampled 1 to 0.0001, trained noise t:{t_val}, eps {eps} Noise estimate:', noise5)

  #sample 2 noise
  print("sample 2 noise estimates:")

  img6_noise = cv2.imread(
    f'/home/christina/Downloads/images/Sampled 2 to eps, trained noise t idx:{noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}.png')
  img_gray6 = cv2.cvtColor(img6_noise, cv2.COLOR_BGR2GRAY)
  noise6 = estimate_noise(img_gray6)
  print(f'Sampled 2 to eps {eps}, trained noise t:{t_val}, eps {eps} Noise estimate:', noise6)

  img7_noise = cv2.imread(
    f'/home/christina/Downloads/images/Sampled 2 to noise t, trained noise t idx: {noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}.png')
  img_gray7 = cv2.cvtColor(img7_noise, cv2.COLOR_BGR2GRAY)
  noise7 = estimate_noise(img_gray7)
  print(f'Sampled 2 to noise t {t_val}, trained noise t:{t_val}, eps {eps} Noise estimate:', noise7)

  img8_noise = cv2.imread(
    f'/home/christina/Downloads/images/Sampled 2 to point zero one, trained noise t idx: {noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}.png')
  img_gray8 = cv2.cvtColor(img8_noise, cv2.COLOR_BGR2GRAY)
  noise8 = estimate_noise(img_gray8)
  print(f'Sampled 2 to 0.01, trained noise t:{t_val}, eps {eps} Noise estimate:', noise8)

  img9_noise = cv2.imread(
    f'/home/christina/Downloads/images/Sampled 2 to point zero zero one, trained noise t idx: {noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}.png')
  img_gray9 = cv2.cvtColor(img9_noise, cv2.COLOR_BGR2GRAY)
  noise9 = estimate_noise(img_gray9)
  print(f'Sampled 2 to 0.001, trained noise t:{t_val}, eps {eps} Noise estimate:', noise9)

  img10_noise = cv2.imread(
    f'/home/christina/Downloads/images/Sampled 2 to point zero zero zero one, trained noise t idx:{noise_values.index(t_val)}, eps idx: {eps_values.index(eps)}.png')
  img_gray10 = cv2.cvtColor(img10_noise, cv2.COLOR_BGR2GRAY)
  noise10 = estimate_noise(img_gray10)
  print(f'Sampled 2 to 0.0001, trained noise t:{t_val}, eps {eps} Noise estimate:', noise10)
