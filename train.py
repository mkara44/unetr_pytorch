from tqdm import tqdm

# Additional Scripts
from utils.utils import EpochCallback, get_dataloader

from config import cfg

from train_unetr import UneTRSeg


class TrainTestPipe:
    def __init__(self, mode=None, dataset_path=None, model_path=None, device=None):
        self.device = device
        self.model_path = model_path

        if mode == "train":
            self.train_loader = get_dataloader(dataset_path, train=True)

        self.val_loader = get_dataloader(dataset_path, train=False)
        self.unetr = UneTRSeg(self.device)

    def __loop(self, loader, step_func, t):
        total_loss = 0

        for step, data in enumerate(loader):
            image, label = data['image'], data['label']
            image = image.to(self.device)
            label = label.to(self.device)

            loss, cls_pred = step_func(image=image, label=label)

            total_loss += loss
            t.update()

        return total_loss

    def train(self):
        callback = EpochCallback(self.model_path, cfg.epoch,
                                 self.unetr.model, self.unetr.optimizer, 'val_loss', cfg.patience)

        for epoch in range(cfg.epoch):
            with tqdm(total=len(self.train_loader) + len(self.val_loader)) as t:
                train_loss = self.__loop(self.train_loader, self.unetr.train_step, t)

                val_loss = self.__loop(self.val_loader, self.unetr.val_step, t)

            callback.epoch_end(epoch + 1,
                               {'loss': train_loss / len(self.train_loader),
                                'val_loss': val_loss / len(self.val_loader)})

            if callback.end_training:
                break

        print("Evaluating...")
        self.evaluate()

    def evaluate(self):
        self.unetr.load_model(self.model_path)
        with tqdm(total=len(self.val_loader)) as t:
            _ = self.__loop(self.val_loader, self.unetr.eval_step, t)

        dice_metric = self.unetr.metric.aggregate()
        print(f"TC Dice coefficient: {round(dice_metric[0].item(), 2)}")
        print(f"WT Dice coefficient: {round(dice_metric[1].item(), 2)}")
        print(f"ET Dice coefficient: {round(dice_metric[2].item(), 2)}")
