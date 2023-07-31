import argparse

import sentencepiece as spm

parser = argparse.ArgumentParser(
    prog='Sentencepiece tokenizer',
    description='This module is used to train sentencepiece tokenizer'
)
parser.add_argument(
    '--file_name',
    type=str,
    help='path to source data which tokenizer will be trained on',
    default='materials/sentences_nlfi.txt',
    required=False)
parser.add_argument(
    '--artifacts_dir',
    type=str,
    help='path to store trained model and vocab',
    default='spm_artifacts',
    required=False)
parser.add_argument(
    '--vocab_size',
    type=int,
    help='size of vocabulary of the model',
    default=512,
    required=False)
parser.add_argument(
    '--model_type',
    type=str,
    help='type of the algorythm to train model',
    default='bpe',
    required=False)
parser.add_argument('--pad_id', type=int, default=0, required=False)
parser.add_argument('--unk_id', type=int, default=1, required=False)
parser.add_argument('--bos_id', type=int, default=2, required=False)
parser.add_argument('--eos_id', type=int, default=3, required=False)

args = parser.parse_args()

SP_MODEL_PREFIX = f'{args.artifacts_dir}/sp_{args.model_type}_{args.vocab_size}'

sp_train_command = f'--input={args.file_name} --model_prefix={SP_MODEL_PREFIX}' \
                   f' --vocab_size={args.vocab_size} --model_type={args.model_type}' \
                   f' --pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id}' \
                   f' --eos_id={args.eos_id}'

spm.SentencePieceTrainer.train(sp_train_command)
