from datetime import datetime

import hydra
import lightning as L
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from datasets.dataset_nlfi import DatasetNlfi
from models.transformer_lm import TransformerLM
from utils.data_utils import collate_fn_padding_offseted_targets


@hydra.main(config_path="../config", config_name="transformer_config")
def train_transformer(cfg: DictConfig):
    # Seed everything
    seed = 228
    np.random.seed(seed)
    L.seed_everything(seed)

    # Initializing model
    transformer_lm = TransformerLM(
        f"{cfg.model.tokenizer.dir}/{cfg.model.tokenizer.model_prefix}.model",
        cfg.model.hidden_dim,
        cfg.trainer.learning_rate,
        cfg.train.dataset.n_tokens,
    )

    # Initializing data
    # train_dataset = DatasetNBlock(cfg.train.dataset.data, transformer_lm.tokenizer, cfg.train.dataset.n_tokens)
    # eval_dataset = DatasetNBlock(cfg.eval.dataset.data, transformer_lm.tokenizer, cfg.train.dataset.n_tokens)

    train_dataset = DatasetNlfi(cfg.train.dataset.data, transformer_lm.tokenizer, cfg.train.dataset.n_lines)
    eval_dataset = DatasetNlfi(cfg.eval.dataset.data, transformer_lm.tokenizer, cfg.eval.dataset.n_lines)

    # -------------------------------------------ВОТ ТУТ Я НАЧАЛ КОПИПАСТИТЬ-------------------------------------------
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.dataloader.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn_padding_offseted_targets(x, transformer_lm.tokenizer.pad_id(),
                                                                 max_seq_len=cfg.train.dataset.n_tokens),
        pin_memory=True,
        # drop_last=True,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.eval.dataloader.batch_size,
        collate_fn=lambda x: collate_fn_padding_offseted_targets(x, transformer_lm.tokenizer.pad_id(),
                                                                 max_seq_len=cfg.train.dataset.n_tokens),
        shuffle=False,
        drop_last=False,
    )

    # Initializing callbacks
    date = datetime.now().strftime("%d.%m.%y_%H.%M")
    version_name = (
        f"{cfg.loggers.tensorboard.subdir}_"
        f"8b_{cfg.trainer.learning_rate}"
        f"lr_{cfg.model.hidden_dim}"  #
        f"hd_{date}"
    )

    logger = TensorBoardLogger(
        save_dir=cfg.loggers.tensorboard.dir,
        name=cfg.loggers.tensorboard.subdir,
        log_graph=True,
        version=version_name,
        default_hp_metric=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.loggers.tensorboard.dir}/{cfg.loggers.tensorboard.subdir}/{version_name}",
        filename="trans_{epoch:2d}_{eval_loss:0.2f}_{eval_pp:0.2f}",
        monitor=cfg.model.checkpoint.metric,
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

    # Training
    trainer = L.Trainer(
        # max_epochs=cfg.trainer.max_epoch, logger=logger, callbacks=[checkpoint_callback]
        max_epochs=cfg.trainer.max_epoch, logger=logger, callbacks=[rich_progress_bar_callback, checkpoint_callback]
    )
    trainer.fit(transformer_lm, train_dataloader, eval_dataloader)

    # Text generation and logging
    transformer_lm = TransformerLM.load_from_checkpoint(checkpoint_callback.best_model_path)
    # sample = eval_dataset[random.randint(2, len(eval_dataset) - 1)][0]
    # input_sample = transformer_lm.tokenizer.decode_ids(sample[:5])
    # truncated_input_sample = input_sample[:5]
    samples = ["Я один в", "Трепещу,", "С колпаком", "Сон ленивый, ", "Почитать меня"]

    generate_sequence(cfg, transformer_lm, version_name, samples, 128)


def generate_sequence(cfg, model, version_name, samples, max_tokens):
    logs = []
    for sample in samples:
        sample_generate_log = f"Input: {sample}\n" f"Output: {model.generate_sequence(sample, max_tokens)}"
        logs.append(sample_generate_log)

    for log in logs:
        print(log)
        print()

    sample_generate_log_file_name = (
        f"{cfg.loggers.tensorboard.dir}/"
        f"{cfg.loggers.tensorboard.subdir}/{version_name}/best_sample_generate_log.txt"
    )

    sample_generate_log_file_mode = "w"

    with open(sample_generate_log_file_name, sample_generate_log_file_mode, encoding="UTF-8") as file:
        for log in logs:
            file.write(f"{log}\n")


if __name__ == "__main__":
    train_transformer()  # noqa pylint: disable=no-value-for-parameter
