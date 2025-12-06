import timm
import torch
import torch.nn as nn


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME, pretrained=True, num_classes=0
        )

        self.mass_mean = config.MASS_MEAN
        self.mass_std = config.MASS_STD

        self.ingr_embed = nn.Embedding(
            config.NUM_INGR + 1, config.EMB_INGR, padding_idx=0
        )
        conc_shape = self.image_model.num_features + 2 * config.EMB_INGR + 1

        self.regressor = nn.Sequential(
            nn.Linear(conc_shape, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, image, ingr_idxs, mass):
        img_emb = self.image_model(image)
        ingr_emb = self.ingr_embed(ingr_idxs)
        ingr_mask = (ingr_idxs != 0).unsqueeze(-1)
        ingr_emb_sum = (ingr_emb * ingr_mask).sum(dim=1)
        ingr_emb_mean = ingr_emb_sum / ingr_mask.sum(dim=1).clamp(min=1)
        # как вариант можно взять среднее?! / ingr_mask.sum(dim=1).clamp(min=1)

        mass_norm = (mass - self.mass_mean) / self.mass_std
        x = torch.cat(
            [img_emb, ingr_emb_sum, ingr_emb_mean, mass_norm.unsqueeze(-1)],
            dim=1,
        )
        return self.regressor(x).squeeze(-1)
