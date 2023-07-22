import os

import lightning as L
import numpy as np
import sentencepiece as spm
import torch
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader

import data_processing
import grulm

seed = 228
np.random.seed(seed)

sp_model_prefix = 'm'
sp = spm.SentencePieceProcessor()
sp.load(f'{sp_model_prefix}.model')

# train_sentences, eval_sentences = train_test_split(data_processing.sentences_8lfi, test_size=0.5,
#
#                                                    random_state=seed)
train_dataset = data_processing.SpDataset(data_processing.sentences_8lfi, sp)
eval_dataset = data_processing.SpDataset(data_processing.sentences_8lfi, sp)

# train_dataset = data_processing.SpDataset(train_sentences, sp)
# eval_dataset = data_processing.SpDataset(eval_sentences, sp)

batch_size = 1

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=data_processing.collate_fn_padding_targets)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size,
                             collate_fn=data_processing.collate_fn_padding_targets)

hidden_dim = 512
grulm = grulm.GRULM(hidden_dim, sp.vocab_size(), sp.pad_id())
logger = TensorBoardLogger("tb_logs", name="grulm", log_graph=True)
trainer = L.Trainer(max_epochs=60,
                    logger=logger,
                    callbacks=RichProgressBar(
                        theme=RichProgressBarTheme(description="s",
                                                   # progress_bar="s",
                                                   # progress_bar_finished="s",
                                                   progress_bar_pulse="rgb(175,0,255)",
                                                   # batch_progress="s",
                                                   time="rgb(175,0,255)",
                                                   # processing_speed="s",
                                                   metrics="rgb(175,0,255)")),
                    overfit_batches=2
                    )
trainer.fit(grulm, train_dataloader, eval_dataloader)

sample = torch.LongTensor(train_dataset[0])
print(f'sample {sample} shape {sample.shape}')
out = grulm(sample)
out = [x.argmax().item() for x in out]
print(f'АУТ ПРОСТО ТАК {sp.decode_ids(out)}')
input_sample = sp.decode_ids(sample.tolist()[:-15])
print(f"А ЭТО ПРИМЕР ГЕНЕРАЦИИ"
      f"\n INPUT :{input_sample} "
      f"\n GENERATIA"
      f" {data_processing.generate_sequence(grulm, input_sample, 512, sp)}")
# print(f'Токены {}')
print('ЛОГИ: ')
os.system('tensorboard --logdir tb_logs/grulm')
# Идинна,,ный,коймкш,ез, в ввый, сра, Иодня,,ел,вый,чирою,хурнакойся " встра, Иесли жна,
# вес сныйя,авиц,еця,дин чторыный,ами Иметжу, и,но,,меною,, вренной,ни,лаго сный,ем, иемет,,
# веняет,, argparse
