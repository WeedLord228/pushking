import torch
import torch.nn as nn

from sp_lightning_module import SpLightningModule


class GRULM(SpLightningModule):
    def __init__(self, hidden_dim, sp_tokenizer_file_name, learning_rate):
        super().__init__(sp_tokenizer_file_name)
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.embedding = nn.Embedding(self.sp.vocab_size(), hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.head = nn.Linear(hidden_dim, self.sp.vocab_size())

        self.loss = nn.CrossEntropyLoss(ignore_index=self.sp.pad_id())
        self.example_input_array = torch.ones(228, dtype=torch.int32)

    def forward(self, x):
        x = self.embedding(x)
        out, final_state = self.rnn(x)
        return self.head(out)

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = self.embedding(x)
        out, final_state = self.rnn(x)

        logits = self.head(out).flatten(start_dim=0, end_dim=1)
        loss = self.loss(logits, y.flatten())

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        out = self(x).flatten(start_dim=0, end_dim=1)
        loss = self.loss(out, y.flatten())

        perplexity = torch.exp(loss).item()
        self.log_dict({'eval_loss': loss, 'eval_pp': perplexity}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def generate_sequence(self, sample, max_len):
        tokenized_sample = torch.LongTensor(
            [self.sp.bos_id()] + (self.sp.encode_as_ids(sample))) if isinstance(
            sample,
            str) else sample

        result_ids = tokenized_sample.tolist()

        next_word = self(tokenized_sample)[-1].argmax()
        result_ids.append(next_word.item())
        c = 1

        while c < max_len and next_word.item() is not self.sp.eos_id():
            tokenized_sample = torch.cat([tokenized_sample, next_word.unsqueeze(0)])
            next_word = self(tokenized_sample)[-1].argmax()
            result_ids.append(next_word.item())
            c = c + 1

        return self.sp.decode_ids(result_ids)
