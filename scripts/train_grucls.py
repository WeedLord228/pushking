from datetime import datetime

import hydra
import lightning as L
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from datasets.govno_dataset import GovnoDataset
from models.grucls import GRUCLS
from utils.data_utils import collate_govno


@hydra.main(config_path="../config", config_name="grucls_train_config")
def train_grucls(cfg: DictConfig):
    seed = 228
    np.random.seed(seed)

    L.seed_everything(seed)

    grulm = GRUCLS(
        cfg.model.hidden_dim,
        f"{cfg.model.tokenizer.dir}/{cfg.model.tokenizer.model_prefix}.model",
        cfg.trainer.learning_rate,
        cls_count=2,
        add_noise=cfg.model.noise,
        adversarial_training=cfg.model.adversarial_training.available,
        adversarial_epsilon=cfg.model.adversarial_training.epsilon,
        virtual_adversarial_training=cfg.model.virtual_adversarial_training.available,
        vat_epsilon=cfg.model.virtual_adversarial_training.epsilon,
        vat_xi=cfg.model.virtual_adversarial_training.xi,
        vat_iterations=cfg.model.virtual_adversarial_training.num_iterations,
    )
    train_dataset = GovnoDataset(cfg.train.dataset.data, grulm.tokenizer)
    eval_dataset = GovnoDataset(cfg.eval.dataset.data, grulm.tokenizer)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.dataloader.batch_size,
        collate_fn=lambda x: collate_govno(x, grulm.tokenizer.pad_id()),
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.eval.dataloader.batch_size,
        collate_fn=lambda x: collate_govno(x, grulm.tokenizer.pad_id()),
        shuffle=False,
        drop_last=False,
    )
    date = datetime.now().strftime("%d.%m.%y_%H.%M")

    vat_name_part = (
        f"_VAT_"
        f"iter{cfg.model.virtual_adversarial_training.num_iterations}_"
        f"eps{cfg.model.virtual_adversarial_training.epsilon}_"
        f"xi{cfg.model.virtual_adversarial_training.xi}__"
    )

    at_name_part = f"_AD_" f"eps{cfg.model.adversarial_training.epsilon}__"

    version_name = (
        f"{cfg.loggers.tensorboard.subdir}_"
        f"lr{cfg.trainer.learning_rate}_"
        f"hd{cfg.model.hidden_dim}_"
        f"{at_name_part if cfg.model.adversarial_training.available else ''}"
        f"{vat_name_part if cfg.model.virtual_adversarial_training.available else ''}"
        f"{date}"
    )

    logger = TensorBoardLogger(
        save_dir=cfg.loggers.tensorboard.dir,
        name=cfg.loggers.tensorboard.subdir,
        log_graph=False,
        version=version_name,
        default_hp_metric=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.loggers.tensorboard.dir}/{cfg.loggers.tensorboard.subdir}/{version_name}",
        filename="grulm_{epoch:2d}_{eval_loss:0.2f}_{eval_f1:0.2f}",
        monitor=cfg.model.checkpoint.metric,
        mode="max",
    )

    rich_progress_bar_callback = RichProgressBar(
        theme=RichProgressBarTheme(
            description="rgb(197,0,27)",
            progress_bar="rgb(175,0,255)",
            progress_bar_finished="rgb(77,167,73)",
            time="rgb(175,0,255)",
            metrics="rgb(175,175,255)",
        )
    )

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=cfg.trainer.max_epoch,
        logger=logger,
        callbacks=[rich_progress_bar_callback, checkpoint_callback],
    )

    # trainer.fit(grulm, train_dataloader, eval_dataloader)
    trainer.fit(grulm, train_dataloader, eval_dataloader)


if __name__ == "__main__":
    train_grucls()  # noqa pylint: disable=no-value-for-parameter
