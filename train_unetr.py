import torch
from torch.optim import AdamW
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (Activations, Compose, AsDiscrete)

from utils.unetr import UneTR
from config import cfg


class UneTRSeg:
    def __init__(self, device):
        self.device = device

        self.model = UneTR(img_dim=cfg.unetr.img_dim,
                           in_channels=cfg.unetr.in_channels,
                           base_filter=cfg.unetr.base_filter,
                           class_num=cfg.unetr.class_num,
                           patch_size=cfg.unetr.patch_size,
                           embedding_dim=cfg.unetr.embedding_dim,
                           block_num=cfg.unetr.block_num,
                           head_num=cfg.unetr.head_num,
                           mlp_dim=cfg.unetr.mlp_dim,
                           z_idx_list=cfg.unetr.z_idx_list).to(self.device)

        self.metric = DiceMetric(include_background=True, reduction="mean_batch")
        self.criterion = DiceCELoss(sigmoid=True)
        self.optimizer = AdamW(self.model.parameters(),
                               lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)

    def load_model(self, path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.model.eval()

    def train_step(self, **params):
        self.model.train()

        self.optimizer.zero_grad()
        pred_mask = self.model(params['image'])
        loss = self.criterion(pred_mask, params['label'])
        loss.backward()
        self.optimizer.step()

        return loss.item(), pred_mask

    def val_step(self, **params):
        with torch.no_grad():
            pred_mask = self.model(params['image'])
            loss = self.criterion(pred_mask, params['label'])

        return loss.item(), pred_mask

    def eval_step(self, **params):
        post_trans = Compose(
            [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
        )

        with torch.no_grad():
            pred_mask = self.model(params['image'])
            pred_mask = [post_trans(i) for i in decollate_batch(pred_mask)]
            self.metric(y_pred=pred_mask, y=params["label"])

        return 0, pred_mask
