import torch.optim
from torch import nn

from models.sp_lightning_module import SpLightningModule


class TransformerLM(SpLightningModule):
    def __init__(self, sp_tokenizer_file_name, emb_dim, learning_rate, max_seq_len):
        super().__init__(sp_tokenizer_file_name)
        self.max_seq_len = max_seq_len

        self.embeddings = nn.Embedding(self.tokenizer.vocab_size(), emb_dim)
        self.pos_embeddings = nn.Embedding(max_seq_len, emb_dim)
        self.transformer_decoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=4, dropout=0.2, batch_first=True
        )
        self.head = nn.Linear(emb_dim, self.tokenizer.vocab_size())

        self.loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id())
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def forward(self, x):
        batch, time = x.shape

        mask = torch.tril(torch.ones(time, time, device=x.device)) == 1
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))

        token_emb = self.pos_embeddings(x)
        position_emb = self.embeddings(torch.arange(time, device=x.device))
        x = token_emb + position_emb

        x = self.transformer_decoder_layer(src=x, src_mask=mask)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):  # pylint: disable=W0613
        x, y = batch

        logits = self(x)
        logits = logits.flatten(start_dim=0, end_dim=1)
        loss = self.loss(logits, y.flatten())

        self.log("train_loss", loss, logger=True, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=W0613
        x, y = batch

        out = self(x).flatten(start_dim=0, end_dim=1)
        loss = self.loss(out, y.flatten())

        perplexity = torch.exp(loss).item()
        self.log_dict(
            {"eval_loss": loss, "eval_pp": perplexity}, prog_bar=True, logger=True, on_epoch=True, on_step=False
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    @torch.no_grad()
    def generate_sequence(self, sample, tokens_to_generate):
        self.eval()
        tokenized_sample = (
            torch.LongTensor(self.tokenizer.encode_as_ids(sample))
            if isinstance(sample, str)
            else sample
        )

        result_ids = tokenized_sample.tolist()
        tokenized_sample = tokenized_sample.to(self.device)

        for _ in range(tokens_to_generate):
            logits = self(torch.unsqueeze(tokenized_sample, 0))[0][-1]
            next_word = logits.argmax(-1)
            result_ids.append(next_word.item())

            if next_word.item() == self.tokenizer.eos_id():
                break

            tokenized_sample = torch.cat([tokenized_sample, next_word.unsqueeze(0)], dim=0)
        return self.tokenizer.decode_ids(result_ids)
