# Ignore warnings
import warnings

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset


warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)


class SpritesDataset(Dataset):
    def __init__(self, imgs, latents_classes, latents_values, transform=None):
        """
        Args:
            dir (string): Directory containing the dSprites dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imgs = imgs
        self.latents_classes = latents_classes
        self.latents_values = latents_values
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = np.expand_dims(self.imgs[idx].astype(np.float32), 0)
        # sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        if self.transform:
            sample = self.transform(sample)
        return sample, [self.latents_classes[idx], self.latents_values[idx]]


# Pytorch Lightning DataModule for Sprites Dataset
class SpritesDataModule(pl.LightningDataModule):
    def __init__(self, dir, batch_size=64, val_split=0.9, transforms=None, seed=42):
        super().__init__()
        self.dir = dir
        self.filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        self.filepath = f"{self.dir}/{self.filename}"

        self.batch_size = batch_size
        self.val_split = val_split
        self.transforms = transforms
        self.seed = seed

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            dataset_zip = np.load(self.filepath, allow_pickle=True, encoding="bytes")

            imgs = dataset_zip["imgs"]
            latents_classes = dataset_zip["latents_classes"]
            latents_values = dataset_zip["latents_values"]
            self.dataset = SpritesDataset(
                imgs,
                latents_classes,
                latents_values,
                transform=self.transforms.ToTensor() if self.transforms else None,
            )

            # Create data indices for training and validation splits:
            dataset_size = len(self.dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(self.val_split * dataset_size))
            if self.seed:
                np.random.seed(self.seed)
                np.random.shuffle(indices)
            train_indices, val_indices = indices[:split], indices[split:]
            rng.shuffle(val_indices)
            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, val_indices)
            # Create data samplers and loaders:
            # self.train_sampler = SubsetRandomSampler(train_indices)
            # self.val_sampler = SubsetRandomSampler(val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # sampler=self.val_sampler,
        )


if __name__ == "__main__":
    # Test
    dm = SpritesDataModule(dir="dsprites-dataset")
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(len(train_loader))
    print(len(val_loader))
    # checking a single batch
    for batch_idx, (data, _) in enumerate(train_loader):
        print(data.shape)
        break
