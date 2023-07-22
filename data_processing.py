import os.path
import re

import sentencepiece as spm
import torch
from torch.utils.data import Dataset


def replace_in_all_lines(template, replacement, lines):
    result = []
    for line in lines:
        a = re.sub(template, replacement, line)
        if a:
            result.append(a)

    assert len(result) == len(lines)
    return result


with open('materials/examples.txt', encoding='utf-8') as f:
    lines = f.readlines()

# raw_lines = [x for x in lines if x.startswith('\t\t')]
raw_lines = lines
no_dia_lines = replace_in_all_lines(r'&#....', '', raw_lines)
data = replace_in_all_lines(r'\xa0', ' ', no_dia_lines)
# data = [x[2:] for x in data]
sent_length = 8
sentences_8lfi = [''.join(data[i:i + sent_length]) for i in range(0, len(data), sent_length)]

file_name = 'materials/sentences_8lfi.txt'
file_mode = 'w' if os.path.exists(file_name) else 'x'
with open(file_name, file_mode, encoding='UTF-8') as f:
    for sentence in sentences_8lfi:
        f.write(sentence)

sp_train_file_name = 'materials/sentences_8lfi.txt'
sp_model_prefix = 'm'
sp_vocab_size = 512
sp_model_type = 'bpe'
sp_pad_id = 0
sp_unk_id = 1
sp_bos_id = 2
sp_eos_id = 3

sp_train_command = f'--input={sp_train_file_name} --model_prefix={sp_model_prefix}' \
                   f' --vocab_size={sp_vocab_size} --model_type={sp_model_type} ' \
                   f'--pad_id={sp_pad_id} --unk_id={sp_unk_id} --bos_id={sp_bos_id}' \
                   f' --eos_id={sp_eos_id}'

spm.SentencePieceTrainer.train(sp_train_command)


def collate_fn_padding_targets(input_batch, pad_id=sp_pad_id):
    max_sent_len = max([len(x) for x in input_batch])
    result_batch = []

    for sequence in input_batch:
        new_seq = sequence
        if len(sequence) < max_sent_len:
            new_seq.extend([pad_id] * (max_sent_len - len(sequence)))
        result_batch.append(new_seq)

    # Простой способ получить таргеты. Те же самые токены, но со сдвигом 1.
    #     result_batch = {
    #         'input'  : result_batch[:,:-1],
    #         'targets': result_batch[:,1:]
    #     }

    result_batch = torch.LongTensor(result_batch)
    return result_batch[:, :-1], result_batch[:, 1:]


class SpDataset(Dataset):
    def __init__(self, sentences, sp_processor):
        self.sentences = sentences
        self.unk_id = sp_processor.unk_id()
        self.bos_id = sp_processor.bos_id()
        self.eos_id = sp_processor.eos_id()
        self.pad_id = sp_processor.pad_id()
        self.sp_processor = sp_processor

    def __getitem__(self, idx):
        result = [self.bos_id]
        result.extend(self.sp_processor.encode_as_ids(self.sentences[idx]))
        result.append(self.eos_id)
        return result

    def __len__(self):
        return len(self.sentences)


def generate_sequence(model, sample, max_len, sp):
    tokenized_sample = torch.LongTensor([sp.bos_id()] + (sp.encode_as_ids(sample))) if isinstance(
        sample,
        str) else sample

    result_ids = tokenized_sample.tolist()

    next_word = model(tokenized_sample)[-1].argmax()
    result_ids.append(next_word.item())
    c = 1

    while c < max_len and next_word.item() is not sp.eos_id():
        tokenized_sample = torch.cat([tokenized_sample, next_word.unsqueeze(0)])
        next_word = model(tokenized_sample)[-1].argmax()
        result_ids.append(next_word.item())
        c = c + 1

    return sp.decode_ids(result_ids)