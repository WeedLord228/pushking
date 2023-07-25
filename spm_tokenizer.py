import sentencepiece as spm

# TODO argparse
SP_INPUT_FILE_NAME = 'materials/sentences_nlfi.txt'
SPM_ARTIFACTS_DIR = 'spm_artifacts'
SP_VOCAB_SIZE = 512
SP_MODEL_TYPE = 'bpe'
SP_MODEL_PREFIX = f'{SPM_ARTIFACTS_DIR}/sp_{SP_MODEL_TYPE}_{SP_VOCAB_SIZE}'
SP_PAD_ID = 0
SP_UNK_ID = 1
SP_BOS_ID = 2
SP_EOS_ID = 3

sp_train_command = f'--input={SP_INPUT_FILE_NAME} --model_prefix={SP_MODEL_PREFIX}' \
                   f' --vocab_size={SP_VOCAB_SIZE} --model_type={SP_MODEL_TYPE}' \
                   f' --pad_id={SP_PAD_ID} --unk_id={SP_UNK_ID} --bos_id={SP_BOS_ID}' \
                   f' --eos_id={SP_EOS_ID}'

spm.SentencePieceTrainer.train(sp_train_command)
