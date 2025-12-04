import os
import random
import torch
import timm
import pandas as pd
from PIL import Image
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from tqdm import tqdm
from torchmetrics.regression import MeanAbsoluteError
from dataset import MultimodalDataset, get_transforms, collate_fn


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def print_gpu_memory():
    '''контроль загрузки видеопамяти'''
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory /1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3 
            reserved = torch.cuda.memory_reserved(i) / 1024**3 
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            utilization = allocated / total_memory * 100
            max_utilization = max_allocated / total_memory * 100
            print(f'\nGPU {i}====>')
            print(f'total mem    : {total_memory:.2f} GB')
            print(f'!current mem : {allocated:.2f} GB ({utilization:.1f}%)')
            print(f'reserved mem : {reserved:.2f} GB')
            print(f'!max per epoch: {max_allocated:.2f} GB ({max_utilization:.1f}%)')
            print(f'=====>GPU {i}\n')
            torch.cuda.reset_peak_memory_stats()
    else:
        print('cuda не доступна')

def plot_training_history(train_losses, val_losses, train_maes, val_maes):

    """
    Построение графиков обучения
    """
    print("\n" + "="*100)
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # loss
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='train Loss', alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='val loss', alpha=0.8)
    ax1.set_xlabel('epoch', fontsize=12)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_ylabel('loss', fontsize=12)
    ax1.set_title('train/val loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    #best val loss
    best_epoch_loss = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    ax1.axvline(x=best_epoch_loss, color='r', linestyle='--', alpha=0.5, 
                label=f'best epoch: {best_epoch_loss}')
    ax1.scatter(best_epoch_loss, best_val_loss, color='r', s=100, zorder=5)
    ax1.legend(fontsize=11)
    
    # MAE
    ax2 = axes[1]
    ax2.plot(epochs, train_maes, 'b-', linewidth=2, label='train MAE', alpha=0.8)
    ax2.plot(epochs, val_maes, 'r-', linewidth=2, label='val MAE', alpha=0.8)
    ax2.set_xlabel('epoch', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_title('train/val MAE', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # best val MAE
    best_epoch_mae = np.argmin(val_maes) + 1
    best_val_mae = min(val_maes)
    ax2.axvline(x=best_epoch_mae, color='r', linestyle='--', alpha=0.5,
                label=f'best epoch: {best_epoch_mae}')
    ax2.scatter(best_epoch_mae, best_val_mae, color='r', s=100, zorder=5)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.show()

    print("\n" + "="*50)
    print("stat:")
    print("="*50)
    print(f"final train loss: {train_losses[-1]:.1f}")
    print(f"final val loss:   {val_losses[-1]:.1f}")
    print(f"final train MAE:  {train_maes[-1]:.1f}")
    print(f"final val MAE:    {val_maes[-1]:.1f}")
    print(f"\nbest val loss:  {best_val_loss:.1f} (epoch {best_epoch_loss})")
    print(f"best val MAE:     {best_val_mae:.1f} (epoch {best_epoch_mae})")
    print("\n" + "="*100)


def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.mass_mean = config.MASS_MEAN
        self.mass_std = config.MASS_STD

        self.ingr_embed = nn.Embedding(config.NUM_INGR + 1, config.EMB_INGR, padding_idx=0)
        conc_shape = self.image_model.num_features + config.EMB_INGR + 1
        
        # self.regressor = nn.Sequential(
        #     nn.Linear(conc_shape, conc_shape // 2),
        #     nn.LayerNorm(conc_shape // 2),
        #     nn.ReLU(),
        #     nn.Dropout(config.DROPOUT),
        #     nn.Linear(conc_shape // 2, 1)
        # )

        self.regressor = nn.Sequential(
            nn.Linear(conc_shape, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, 1)
        )

    def forward(self, image, ingr_idxs, mass):
        img_emb = self.image_model(image)
        ingr_emb = self.ingr_embed(ingr_idxs)
        ingr_mask = (ingr_idxs != 0).unsqueeze(-1) 
        ingr_emb = (ingr_emb * ingr_mask).sum(dim=1) #/ ingr_mask.sum(dim=1).clamp(min=1)

        mass_norm = (mass - self.mass_mean) / self.mass_std 
        x = torch.cat([img_emb, ingr_emb, mass_norm.unsqueeze(-1)], dim=1)
        return self.regressor(x).squeeze(-1)    

def validate(model, val_loader, epoch_index, criterion, device):
    model.eval()
    epoch_mae = MeanAbsoluteError().to(device)
    batch_mae = MeanAbsoluteError().to(device)
    total_loss = 0.0
    samples = 0.0

    progress_bar = tqdm(
                val_loader,
                desc=f"{'  val epoch ' + str(epoch_index+1):<15}",
            )

    with torch.no_grad():
        for batch in progress_bar:
            batch_mae.reset()
            inputs = {
                'image': batch['image'].to(device),
                'ingr_idxs': batch['ingr_idxs'].to(device),
                'mass': batch['mass'].to(device)
            }
            labels = batch['label'].to(device)

            predicted = model(**inputs)
            loss = criterion(predicted, labels)
            total_loss += loss.item()*len(labels)
            samples += len(labels)
            
            epoch_mae.update(predicted, labels)    
            batch_mae.update(predicted, labels)    
            val_batch_mae = batch_mae.compute().item()
            progress_bar.set_postfix(
            {
                'last batch loss': f'{loss.item():.1f}',
                'last batch mae': f'{val_batch_mae:.1f}'
             }
             )
    
    val_epoch_mae = epoch_mae.compute().item()
    return  val_epoch_mae, total_loss / samples

def train_one_epoch(model, train_loader, val_loader, device, optimizer, criterion, epoch_index=0):
    model.train()
    epoch_mae = MeanAbsoluteError().to(device)
    batch_mae = MeanAbsoluteError().to(device)
    total_loss = 0.0
    samples = 0.0

    progress_bar = tqdm(
                train_loader,
                desc=f"{'train epoch ' + str(epoch_index+1):<15}",
            )

    for batch in progress_bar:
        batch_mae.reset()
        inputs = {
            'image': batch['image'].to(device),
            'ingr_idxs': batch['ingr_idxs'].to(device),
            'mass': batch['mass'].to(device)
        }
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        predicted = model(**inputs)
        loss = criterion(predicted, labels)

        total_loss += loss.item()*len(labels)
        samples += len(labels)
        epoch_mae.update(predicted, labels)
        batch_mae.update(predicted, labels)
        train_batch_mae = batch_mae.compute().item()

        progress_bar.set_postfix(
            {
                'last batch loss': f'{loss.item():.1f}',
                'last batch mae': f'{train_batch_mae:.1f}'
             }
             )
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    train_epoch_loss = total_loss / samples
    train_epoch_mae = epoch_mae.compute().item()
    val_epoch_mae, val_epoch_loss = validate(model,val_loader, epoch_index, criterion, device)
    
    # print_gpu_memory() # отключаем для компактного лога
    
    print(f"{'total epoch ' + str(epoch_index+1):<15}", f'train loss {train_epoch_loss:.1f}, val loss {val_epoch_loss:.1f}', f'train mae {train_epoch_mae:.1f}, val mae {val_epoch_mae:.1f}')
    print('-'*40)
    return train_epoch_loss, train_epoch_mae, val_epoch_loss, val_epoch_mae

def get_loaders(config):
    transforms = get_transforms(config, ds_type="train")
    val_transforms = get_transforms(config, ds_type="test")
    train_dataset = MultimodalDataset(config, transforms, ds_type="train")
    val_dataset = MultimodalDataset(config, val_transforms, ds_type="test")
    train_loader = DataLoader(train_dataset,
                                batch_size=config.BATCH_SIZE,
                                shuffle=True,
                                collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset,
                                batch_size=config.BATCH_SIZE,
                                shuffle=False,
                                collate_fn=collate_fn)

    return train_loader, val_loader

def train(config, device):
        
    seed_everything(config.SEED)
    model = MultimodalModel(config).to(device)

    set_requires_grad(model.image_model, config.IMAGE_MODEL_UNFREEZE)

    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params to train: {train_params:,}")

    optimizer = AdamW([{
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, {
        'params': model.ingr_embed.parameters(),
        'lr': config.EMB_INGR_LR
    }, {
        'params': model.regressor.parameters(),
        'lr': config.CLASSIFIER_LR
    }]
    ,weight_decay=config.WEIGHT_DECAY
    )

    criterion = nn.HuberLoss(delta=config.HUBER_DELTA)

    train_loader, val_loader = get_loaders(config)
    
    train_loss_stat, val_loss_stat, train_mae_stat, val_mae_stat = [], [], [], []
    best_mae_val = np.inf
    for i in range(config.EPOCHS):
        (train_epoch_loss,
         train_epoch_mae,
          val_epoch_loss,
          val_epoch_mae
         ) = train_one_epoch(model, train_loader, val_loader, device, optimizer, criterion, epoch_index=i)
        if val_epoch_mae < best_mae_val:
            print(f"New best model, epoch: {i+1}")
            print('-'*40)
            best_mae_val = val_epoch_mae
            torch.save(model.state_dict(), config.SAVE_PATH)
        train_loss_stat.append(train_epoch_loss)
        val_loss_stat.append(val_epoch_loss)
        train_mae_stat.append(train_epoch_mae)
        val_mae_stat.append(val_epoch_mae)
        if (i + 1) % 9 == 0:
            plot_training_history(train_loss_stat, val_loss_stat, train_mae_stat, val_mae_stat)
    return train_loss_stat, val_loss_stat, train_mae_stat, val_mae_stat    

def check_inference(config, device):

    def _plot_preds(df, num, best=False):
        ig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        df = df.sort_values(by='mae', ascending=best)
        for i in range(num):
            row = df.iloc[i]
            ax = axes[i]
            
            # Загружаем изображение
            img = Image.open(row['img']).convert("RGB")
            ax.imshow(img)
            
            # Подпись: mass, pred, target, mae
            ax.set_title(
                f"mass: {row['mass']:.0f}\n"
                f"pred: {row['pred']:.1f} | true: {row['target']:.1f}\n"
                f"MAE: {row['mae']:.1f}",
                fontsize=10, 
                ha='center'
            )
            ax.axis('off')
        title = "лучших" if best else "худших"
        plt.suptitle(f"TOP-{num} {title} предсказаний", fontsize=12, y=0.98)
        plt.tight_layout()
        plt.show()
    
    _, val_loader = get_loaders(config)
    model = MultimodalModel(config)
    model.load_state_dict(torch.load(config.SAVE_PATH,  map_location=device))
    model.to(device) 
    model.eval()
    img_paths =[]
    masses=[]
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                    'image': batch['image'].to(device),
                    'ingr_idxs': batch['ingr_idxs'].to(device),
                    'mass': batch['mass'].to(device)
            }
            img_path = batch['img_path']
            labels = batch["label"]

            preds = model(**inputs)
            
            img_paths.extend(img_path)
            masses.extend(batch['mass'].tolist())
            predictions.extend(preds.cpu().tolist())
            targets.extend(labels.tolist())
    
    data = list(zip(img_paths, masses, predictions, targets))
    df = pd.DataFrame(data, columns=['img', 'mass', 'pred', 'target'])
    df['mae'] = abs(df['pred'] - df['target'])
    print('='*20)
    print(f'val MAE: {df["mae"].mean():.1f}')
    print('='*20)
    _plot_preds(df, 6, best=False)
    _plot_preds(df, 6, best=True)