from diffusers import DDPMScheduler
from diffusers import UNet2DModel
import torch
import PIL.Image
import numpy as np
from torchvision import utils as vutils
import matplotlib.pyplot as plt
import tqdm

from dataclasses import dataclass
from datasets import load_dataset
from torchvision import transforms
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator

from tqdm.auto import tqdm
from pathlib import Path
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def show(clean_img, sample, epoch, i):

    img1 = clean_img.cpu().squeeze().permute(1, 2, 0)
    # img = (img + 1.0) * 127.5
    img1 = (img1 * 0.5 + 0.5) * 255
    img1 = img1.numpy().astype('uint8')
    img2 = sample.cpu().squeeze().permute(1, 2, 0)
    # img = (img + 1.0) * 127.5
    img2 = (img2 * 0.5 + 0.5) * 255
    img2 = img2.numpy().astype('uint8')
    # print(img.shape)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img1.astype('uint8'))
    axs[0].set_axis_off()
    axs[1].imshow(img2.astype('uint8'))
    axs[1].set_axis_off()
    plt.title("step " + str(epoch) + " i " + str(i))
    plt.savefig(os.path.join("./imgs_fold", "epoch-" + str(epoch) + "_i-" + str(i)+".jpg"))
    plt.show()

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 1
    eval_batch_size = 1  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'ddpm-flowers-256'  # the model namy locally and on the HF Hub

    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()

config.dataset_name = r"E:\datasets\flowers"
dataset = load_dataset(config.dataset_name, split="train")

print(dataset)

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(dataset[:4]["image"]):
    axs[i].imshow(image)
    axs[i].set_axis_off()
fig.show()

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(dataset[:4]["images"]):
    image = image.permute(1, 2, 0).numpy()
    image = (image * 0.5 + 0.5) * 255
    axs[i].imshow(image.astype('uint8'))
    axs[i].set_axis_off()
fig.show()

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# pretrained model
repo_id = "./pretrained"
model = UNet2DModel.from_pretrained(repo_id)

scheduler = DDPMScheduler.from_pretrained(repo_id)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            # After each step you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                sample_noise = torch.randn(clean_images.shape).to(clean_images.device)
                sample = sample_noise

                for i, t in enumerate(scheduler.timesteps):
                    # 1. predict noise residual
                    with torch.no_grad():
                        residual = model(sample, t).sample

                    # 2. compute less noisy image and set x_t -> x_t-1
                    sample = scheduler.step(residual, t, sample).prev_sample

                    # 3. optionally look at image
                    if (i + 1) % 50 == 0:
                        # combine_img = torch.cat((clean_images, sample))
                        show(clean_images, sample, step, i)
                        # vutils.save_image(combine_img, os.path.join('./imgs_fold', f"{epoch}_{step}_{i}.jpg"), nrow=1)


if __name__ == "__main__":
    train_loop(config, model, scheduler, optimizer, train_dataloader, lr_scheduler)





