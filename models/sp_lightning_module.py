import lightning as L
import sentencepiece as spm


class SpLightningModule(L.LightningModule):
    def __init__(self, sp_tokenizer_file_name):
        super().__init__()
        self.tokenizer = spm.SentencePieceProcessor(sp_tokenizer_file_name)  # pylint: disable=E1121
