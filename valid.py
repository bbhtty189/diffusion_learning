from diffusers import DDPMScheduler
from diffusers import UNet2DModel
import torch
import PIL.Image
import numpy as np
from torchvision import utils as vutils
import matplotlib.pyplot as plt
import tqdm

def show(img, i):

    img = img.cpu().squeeze().permute(1, 2, 0)
    # img = (img + 1.0) * 127.5
    img = (img * 0.5 + 0.5) * 255
    img = img.numpy()
    # print(img.shape)
    plt.imshow(img.astype('uint8'))
    plt.title("step "+str(i))
    plt.axis('off')
    plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# pretrained model
repo_id = "./pretrained"
model = UNet2DModel.from_pretrained(repo_id)

print(model.config)

# random noise
torch.manual_seed(0)

noisy_sample = torch.randn(
    1, model.config.in_channels, model.config.sample_size, model.config.sample_size
)
print(noisy_sample.shape)

with torch.no_grad():
    noisy_residual = model(sample=noisy_sample, timestep=2).sample

print(noisy_residual.shape)

scheduler = DDPMScheduler.from_pretrained(repo_id)

print(scheduler.config)

less_noisy_sample = scheduler.step(
    model_output=noisy_residual, timestep=2, sample=noisy_sample
).prev_sample

print(less_noisy_sample.shape)

show(less_noisy_sample, 2)

model.to(device)
noisy_sample = noisy_sample.to(device)

sample = noisy_sample

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
  # 1. predict noise residual
  with torch.no_grad():
      residual = model(sample, t).sample

  # 2. compute less noisy image and set x_t -> x_t-1
  sample = scheduler.step(residual, t, sample).prev_sample

  # 3. optionally look at image
  if (i + 1) % 50 == 0:
      show(sample, i)
