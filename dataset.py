import torch
from torch.utils.data import Dataset

from PIL import Image
import timm
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

import albumentations as A


class MultimodalDataset(Dataset):

    def __init__(self, config, transforms, ds_type="train", initial_img=False):

        df = pd.read_csv(config.DF_PATH)

        # создаем числовую мапу индексов ингредиентов
        ingrs = (
            df[df["split"] == "train"]["ingredients"]
            .str.split(";")
            .explode()
            .sort_values()
            .unique()
        )
        config.NUM_INGR = len(ingrs)
        log_mass = np.log(df[df["split"] == "train"]["total_mass"])
        initial_mass = df[df["split"] == "train"]["total_mass"]
        # config.MASS_MEAN = log_mass.mean()
        # config.MASS_STD = log_mass.std()
        config.MASS_MEAN = initial_mass.mean()
        config.MASS_STD = initial_mass.std()

        self.ingrs_map = {ingr: i + 1 for i, ingr in enumerate(ingrs)}

        if ds_type == "train":
            self.df = df[df["split"] == "train"].reset_index()
        else:
            self.df = df[df["split"] == "test"].reset_index()
        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transforms = transforms
        self.initial_img = initial_img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        ingrs = self.df.loc[idx, "ingredients"].split(";")

        ingr_idxs = [self.ingrs_map.get(i, 0) for i in ingrs]

        initial_mass = self.df.loc[idx, "total_mass"]
        log_mass = np.log(initial_mass)
        mass = initial_mass
        label = self.df.loc[idx, "total_calories"]

        img_path = self.df.loc[idx, "dish_id"]
        image = Image.open(f"data/images/{img_path}/rgb.png").convert("RGB")

        image_transformed = self.transforms(image=np.array(image))["image"]
        item = {
            "img_path": f"data/images/{img_path}/rgb.png",
            "label": torch.tensor(label, dtype=torch.float32),
            "image": image_transformed,
            "ingr_idxs": torch.tensor(ingr_idxs),
            "initital_mass": torch.tensor(initial_mass, dtype=torch.float32),
            "mass": torch.tensor(mass, dtype=torch.float32),
        }
        if self.initial_img:
            item["initial_image"] = image
        return item


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    img_path = [item["img_path"] for item in batch]
    ingr_lists = []
    for item in batch:
        idxs = item["ingr_idxs"]
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.tolist()
        ingr_lists.append(idxs)

    max_len = max(len(idxs) for idxs in ingr_lists)
    padded_ingr = [
        idxs + [0] * (max_len - len(idxs)) for idxs in ingr_lists  # 0 = padding
    ]
    ingr_idxs = torch.tensor(padded_ingr, dtype=torch.long)

    labels = torch.tensor([item["label"] for item in batch])

    mass = torch.tensor([item["mass"] for item in batch])

    return {
        "label": labels,
        "img_path": img_path,
        "image": images,
        "mass": mass,
        "ingr_idxs": ingr_idxs,
    }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0
                ),
                A.Affine(
                    scale=(0.9, 1.1),
                    rotate=(-10, 10),
                    translate_percent=(-0.05, 0.05),
                    shear=(-5, 5),
                    fill=128,
                    p=0.8,
                ),
                A.CoarseDropout(
                    num_holes_range=(2, 8),
                    hole_height_range=(
                        int(0.07 * cfg.input_size[1]),
                        int(0.15 * cfg.input_size[1]),
                    ),
                    hole_width_range=(
                        int(0.1 * cfg.input_size[2]),
                        int(0.15 * cfg.input_size[2]),
                    ),
                    fill=128,
                    p=0.5,
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7
                ),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0),
            ],
            seed=42,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0
                ),
                A.CenterCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0
                ),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0),
            ],
            seed=42,
        )

    return transforms
