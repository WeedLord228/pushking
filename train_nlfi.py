import random
from datetime import datetime

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dataset_nlfi import DatasetNlfi
from grulm import GRULM
from utils import collate_fn_padding_offseted_targets

# TODO hydra
SP_ARTIFACTS_DIR = 'spm_artifacts'
SP_MODEL_PREFIX = 'sp_bpe_512'
TRAIN_DATA_FILE = 'materials/train_nlfi.txt'
EVAL_DATA_FILE = 'materials/eval_nlfi.txt'
N_LINES = 8
BATCH_SIZE = 16
HIDDEN_DIM = 512
MAX_EPOCH = 20
LOG_DIR = 'tb_logs'
LOG_SUBDIR = 'grulm'
LEARNING_RATE = 1e-3

seed = 228
np.random.seed(seed)
L.seed_everything(seed)

grulm = GRULM(HIDDEN_DIM, f'{SP_ARTIFACTS_DIR}/{SP_MODEL_PREFIX}.model', LEARNING_RATE)

train_dataset = DatasetNlfi(TRAIN_DATA_FILE, grulm.sp, N_LINES)
eval_dataset = DatasetNlfi(EVAL_DATA_FILE, grulm.sp, N_LINES)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              collate_fn=lambda x: collate_fn_padding_offseted_targets(x, grulm.sp.pad_id()))
eval_dataloader = DataLoader(dataset=eval_dataset,
                             batch_size=BATCH_SIZE,
                             collate_fn=lambda x: collate_fn_padding_offseted_targets(x, grulm.sp.pad_id()))

date = datetime.now().strftime("%d.%m.%y_%H.%M")
version_name = f'{LOG_SUBDIR}_{N_LINES}l_{LEARNING_RATE}lr_{HIDDEN_DIM}hd_{date}'
logger = TensorBoardLogger(save_dir=LOG_DIR,
                           name=LOG_SUBDIR,
                           log_graph=True,
                           version=version_name)

trainer = L.Trainer(max_epochs=MAX_EPOCH,
                    logger=logger,
                    callbacks=RichProgressBar(
                        theme=RichProgressBarTheme(description="rgb(197,0,27)",
                                                   progress_bar="rgb(175,0,255)",
                                                   progress_bar_finished="rgb(77,167,73)",
                                                   time="rgb(175,0,255)",
                                                   metrics="rgb(175,175,255)"))
                    )
trainer.fit(grulm, train_dataloader, eval_dataloader)

sample = torch.LongTensor(train_dataset[random.randint(2, len(eval_dataloader))][0])
input_sample = grulm.sp.decode_ids(sample.tolist()[:10])
sample_generate_log = f'Original: {grulm.sp.decode_ids(sample.tolist())}\n' \
                      f'Input: {input_sample}\n' \
                      f'Output: {grulm.generate_sequence(input_sample, 512)}'
print(sample_generate_log)

sample_generate_log_file_name = f'{LOG_DIR}/{LOG_SUBDIR}/{version_name}/sample_generate_log.txt'
sample_generate_log_file_mode = 'w'

with open(sample_generate_log_file_name, sample_generate_log_file_mode, encoding='UTF-8') as file:
    file.write(sample_generate_log)
