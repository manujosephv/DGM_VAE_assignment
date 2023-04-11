from typing import List

import pytorch_lightning as pl
import torch
import wandb
import uuid

from datamodule import SpritesDataModule
from simple_vae import ConvVAE

SEED = 42


class ConvBetaVAE(ConvVAE):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        encoder_decoder_dims: List = [32, 64, 128, 256, 512],
        beta=4,
    ):
        super().__init__(
            in_channels=in_channels,
            latent_dim=latent_dim,
            encoder_decoder_dims=encoder_decoder_dims,
        )
        self.beta = beta

    def loss_function(self, recon_x, x, mu, logvar, m_by_n):
        MSE = (recon_x - x).pow(2).sum()
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + self.beta * m_by_n * KLD

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-6)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        input_shape = x.shape[-1]
        x_recon, mu, log_var = self.forward(x)
        loss = self.loss_function(
            x_recon, x, mu, log_var, self.hparams.latent_dim / input_shape
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        input_shape = x.shape[-1]
        x_recon, mu, log_var = self.forward(x)
        loss = self.loss_function(
            x_recon, x, mu, log_var, self.hparams.latent_dim / input_shape
        )
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


def train():
    wandb.login(key="1a6be1c224ac6e7582f1f314f2409a5d403324a0")
    dm = SpritesDataModule(
        dir="dsprites-dataset",
        batch_size=128,
        val_split=0.9,
        transforms=None,
        seed=SEED,
    )
    dm.setup()

    BETA = 3
    model = ConvBetaVAE(
        in_channels=1,
        latent_dim=6,
        encoder_decoder_dims=[64, 128, 256, 512, 1024],
        beta=BETA,
    )
    # Wandb logger
    wandb_logger = pl.loggers.WandbLogger(
        project="VAE", name=f"ConvBeta_{BETA}_VAE_{uuid.uuid4()}"
    )
    es = pl.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename=f"ConvBeta_{BETA}_VAE_{uuid.uuid4()}" + "-{epoch:02d}-{val_loss:.2f}",
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
