from typing import List

import pytorch_lightning as pl
import torch
import wandb
import uuid
from torch import nn

from datamodule import SpritesDataModule
from base_vae import BaseVAE, Flatten, UnFlatten

SEED = 42


class ConvVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        encoder_decoder_dims: List = [32, 64, 128, 256, 512],
    ):
        super().__init__(
            in_channels=in_channels,
            latent_dim=latent_dim,
            encoder_decoder_dims=encoder_decoder_dims,
        )

    def build_layers(self):
        sample_input = torch.randn(1, self.hparams.in_channels, 64, 64)
        modules = []
        in_channels = self.hparams.in_channels
        for dim in self.hparams.encoder_decoder_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                )
            )
            in_channels = dim
        self.encoder = nn.Sequential(*modules)
        self.flatten = Flatten()
        latent = self.flatten(self.encoder(sample_input))
        self.fc_mu = nn.Linear(latent.shape[1], self.hparams.latent_dim)
        self.fc_var = nn.Linear(latent.shape[1], self.hparams.latent_dim)

        # Build Decoder
        self.decoder_input = nn.Linear(self.hparams.latent_dim, latent.shape[1])
        self.unflatten = UnFlatten(in_channels)
        modules = []
        hidden_dims = list(reversed(self.hparams.encoder_decoder_dims))
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                1,
                kernel_size=2,
                stride=2,
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        # var = nn.ELU()(self.fc_var(x))
        return mu, log_var

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.unflatten(x)
        x = self.decoder(x)
        return self.final_layer(x)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructions = self.decode(z)
        return reconstructions, mu, log_var

    def sample(self, num_samples, **kwargs):
        z = torch.randn(num_samples, self.hparams.latent_dim, device=self.device)
        return self.decode(z)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        # eps = torch.randn_like(std)
        # return mu + eps * std
        q = torch.distributions.Normal(mu, std)
        return q.rsample()

    def loss_function(self, recon_x, x, mu, logvar):
        """Reconstruction + KL divergence losses summed over all elements (of a batch)
        see Appendix B from VAE paper:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114
        KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        MSE = (recon_x - x).pow(2).sum()
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-6)


def train():
    wandb.login(key="1a6be1c224ac6e7582f1f314f2409a5d403324a0")
    dm = SpritesDataModule(
        dir="dsprites-dataset",
        batch_size=512,
        val_split=0.9,
        transforms=None,
        seed=SEED,
    )
    dm.setup()
    model = ConvVAE(
        in_channels=1,
        latent_dim=6,
        encoder_decoder_dims=[64, 128, 256, 512, 1024],
    )
    # Wandb logger
    wandb_logger = pl.loggers.WandbLogger(project="VAE", name=f"ConvVAE_{uuid.uuid4()}")
    es = pl.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename=f"ConvVAE_{uuid.uuid4()}" + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=[es, checkpoint_callback],
        max_epochs=50,
        logger=wandb_logger,
        fast_dev_run=False,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    pl.seed_everything(SEED)
    train()
