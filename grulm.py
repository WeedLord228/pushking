import lightning as L
import torch
import torch.nn as nn


class GRULM(L.LightningModule):
    def __init__(self, hidden_dim, vocab_size, pad_id):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.head = nn.Linear(hidden_dim, vocab_size)

        self.loss = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.example_input_array = torch.ones(110, dtype=torch.int32)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
