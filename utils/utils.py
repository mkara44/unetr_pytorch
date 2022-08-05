import os
import torch
import numpy as np

from monai.transforms import (
    Compose,
    Resized,
    ToTensord,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    Spacingd,
    EnsureChannelFirstd,
    RandSpatialCropd,
    RandScaleIntensityd
)
from monai.apps import DecathlonDataset
from monai.data import DataLoader

# Additional Scripts
from config import cfg


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, dim=0).float()
        return d


def get_dataloader(path, train):
    if train:
        shuffle = True
        batch_size = cfg.batch_size
        transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                ToTensord(keys=['image', 'label']),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Resized(keys=['image', 'label'], spatial_size=cfg.unetr.img_dim, mode='nearest'),
                RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )

    else:
        shuffle = False
        batch_size = 1
        transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                ToTensord(keys=['image', 'label']),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Resized(keys=['image', 'label'], spatial_size=cfg.unetr.img_dim, mode='nearest'),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

    download = False if os.path.exists(os.path.join(path, "Task01_BrainTumour")) else True
    print(f"Dataset will be downloaded: {download}")

    dataset = DecathlonDataset(root_dir=path,
                               task="Task01_BrainTumour",
                               transform=transform,
                               section="training" if train else "validation",
                               download=download,
                               cache_rate=0.0)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


class EpochCallback:
    end_training = False
    not_improved_epoch = 0
    monitor_value = np.inf

    def __init__(self, model_name, total_epoch_num, model, optimizer, monitor=None, patience=None):
        if isinstance(model_name, str):
            model_name = [model_name]
            model = [model]
            optimizer = [optimizer]

        self.model_name = model_name
        self.total_epoch_num = total_epoch_num
        self.monitor = monitor
        self.patience = patience
        self.model = model
        self.optimizer = optimizer

    def __save_model(self):
        for m_name, m, opt in zip(self.model_name, self.model, self.optimizer):
            torch.save({'model_state_dict': m.state_dict(),
                        'optimizer_state_dict': opt.state_dict()},
                       m_name)

            print(f'Model saved to {m_name}')

    def epoch_end(self, epoch_num, hash):
        epoch_end_str = f'Epoch {epoch_num}/{self.total_epoch_num} - '
        for name, value in hash.items():
            epoch_end_str += f'{name}: {round(value, 4)} '

        print(epoch_end_str)

        if self.monitor is None:
            self.__save_model()

        elif hash[self.monitor] < self.monitor_value:
            print(f'{self.monitor} decreased from {round(self.monitor_value, 4)} to {round(hash[self.monitor], 4)}')

            self.not_improved_epoch = 0
            self.monitor_value = hash[self.monitor]
            self.__save_model()
        else:
            print(f'{self.monitor} did not decrease from {round(self.monitor_value, 4)}, model did not save!')

            self.not_improved_epoch += 1
            if self.patience is not None and self.not_improved_epoch >= self.patience:
                print("Training was stopped by callback!")
                self.end_training = True


def create_folder_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)
