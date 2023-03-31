from typing import List

import pytorch_lightning as pl
import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels

    def forward(self, input):
        size = int((input.size(1) // self.n_channels) ** 0.5)
        return input.view(input.size(0), self.n_channels, size, size)


class BaseVAE(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        encoder_decoder_dims: List = [32, 64, 128, 256, 512],
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.build_layers()

    def build_layers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def encode(self, x):
        raise NotImplementedError()

    def decode(self, z):
        raise NotImplementedError()

    def sample(self, num_samples, **kwargs):
        raise NotImplementedError()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        q = torch.distributions.Normal(mu, std)
        return q.rsample()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-6)

    def loss_function(self, recon_x, x, mu, logvar):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, mu, log_var = self.forward(x)
        loss = self.loss_function(x_recon, x, mu, log_var)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, mu, log_var = self.forward(x)
        loss = self.loss_function(x_recon, x, mu, log_var)
        self.log("val_loss", loss)
        # Log sample images
        if batch_idx == 0:
            n = min(x.size(0), 8)
            comparison = torch.cat(
                [x[:n], torch.zeros_like(x[:n])[:, :, :, :10], x_recon[:n]], dim=-1
            ).cpu()
            # split into list of images
            comparison = [comparison[i] for i in range(comparison.shape[0])]
            self.logger.log_image("reconstruction", comparison)
