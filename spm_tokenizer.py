import argparse

import sentencepiece as spm

parser = argparse.ArgumentParser(
    prog='Sentencepiece tokenizer',
    description='This module is used to train sentencepiece tokenizer'
)
parser.add_argument(
    '--filename',
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

SP_INPUT_FILE_NAME, SPM_ARTIFACTS_DIR, SP_VOCAB_SIZE, SP_MODEL_TYPE, SP_PAD_ID, SP_UNK_ID, SP_BOS_ID, SP_EOS_ID = \
    args.__reduce__()[2].values()

SP_MODEL_PREFIX = f'{SPM_ARTIFACTS_DIR}/sp_{SP_MODEL_TYPE}_{SP_VOCAB_SIZE}'

sp_train_command = f'--input={SP_INPUT_FILE_NAME} --model_prefix={SP_MODEL_PREFIX}' \
                   f' --vocab_size={SP_VOCAB_SIZE} --model_type={SP_MODEL_TYPE}' \
                   f' --pad_id={SP_PAD_ID} --unk_id={SP_UNK_ID} --bos_id={SP_BOS_ID}' \
                   f' --eos_id={SP_EOS_ID}'

spm.SentencePieceTrainer.train(sp_train_command)
