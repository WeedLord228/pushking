import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from models.sp_lightning_module import SpLightningModule


class GRUCLS(SpLightningModule):
    def __repr__(self):
        return (
            f"AT-{self.adversarial_training}-"
            f"VAT-{self.virtual_adversarial_training}-xi{self.vat_xi}-eps{self.vat_epsilon}-{self.vat_iterations}"
        )

    def __init__(
            self,
            hidden_dim,
            sp_tokenizer_file_name,
            learning_rate,
            cls_count,
            add_noise=False,
            adversarial_training=False,
            adversarial_epsilon=None,
            virtual_adversarial_training=False,
            vat_iterations=None,
            vat_epsilon=None,
            vat_xi=None,
    ):
        super().__init__(sp_tokenizer_file_name)
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.embedding = nn.Embedding(self.tokenizer.vocab_size(), hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.head = nn.Linear(hidden_dim, cls_count)

        self.loss = nn.CrossEntropyLoss()
        self.f1 = torchmetrics.F1Score("binary", num_classes=2, average="macro")

        self.add_noise = add_noise

        self.adversarial_training = adversarial_training
        self.adversarial_epsilon = adversarial_epsilon

        self.virtual_adversarial_training = virtual_adversarial_training
        self.vat_iterations = vat_iterations
        self.vat_epsilon = vat_epsilon
        self.vat_xi = vat_xi

    def forward(self, x, y=None):
        x = self.embedding(x)

        if self.add_noise:
            epsilon = 0.01  # Hyperparameter, adjust as needed
            x = self.add_random_perturbation(x, epsilon)

        if self.adversarial_training and y is not None:
            x = self.compute_adversarial_perturbation(x, y, self.adversarial_epsilon)

        if self.virtual_adversarial_training and y is not None:
            r_vadv = self.compute_vat_perturbation(
                x, xi=self.vat_xi, epsilon=self.vat_epsilon, num_power_iterations=self.vat_iterations
            )
            x += r_vadv

        out, _ = self.rnn(x)
        return self.head(out.mean(-2))

    def training_step(self, batch, batch_idx):  # pylint: disable=W0613
        self.f1.reset()
        x, y = batch

        out = self(x, y)
        loss = self.loss(out, y)
        f1 = self.f1(out.softmax(-1).argmax(-1), y)
        self.log_dict({"train_loss": loss, "train_f1": f1}, logger=True, prog_bar=True, on_epoch=False, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=W0613
        self.f1.reset()
        x, y = batch

        out = self(x)
        loss = self.loss(out, y)
        f1 = self.f1(out.softmax(-1).argmax(-1), y)
        self.log_dict({"eval_loss": loss, "eval_f1": f1}, prog_bar=True, logger=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def generate_answer(self, sample):
        dic = {0: "троленг", 1: "терпила"}
        with torch.no_grad():
            tokenized_sample = (
                torch.LongTensor((self.tokenizer.encode_as_ids(sample)) + [self.tokenizer.eos_id()])
                if isinstance(sample, str)
                else sample
            )

            self.eval()

            tokenized_sample = tokenized_sample.to(self.device)
            next_word = self(tokenized_sample).softmax(-1).argmax(-1)

            return dic[next_word.item()]

    def add_random_perturbation(self, embeddings, epsilon):
        # Adding a small random noise to the embeddings
        noise = torch.randn_like(embeddings) * epsilon
        return embeddings + noise

    def compute_adversarial_perturbation(self, embeddings, y, epsilon):
        embeddings_adv = embeddings.clone().detach().requires_grad_(True)  # Corrected here
        out = self.rnn(embeddings_adv)[0]
        out = self.head(out.mean(1))
        loss = self.loss(out, y)
        self.zero_grad()  # Ensure previous gradients are cleared
        loss.backward(retain_graph=True)

        # Compute adversarial perturbation
        # TODO шумно ходить
        perturbation = epsilon * embeddings_adv.grad.sign()
        return embeddings + perturbation.detach()

    def compute_vat_perturbation(self, embeddings, xi=1e-2, epsilon=1e-2, num_power_iterations=1):
        cooler_embeddings = embeddings.clone()

        # Initialize perturbation d
        d = torch.randn_like(cooler_embeddings, requires_grad=True)
        d = _l2_normalize(d)

        # Power iterations
        for _ in range(num_power_iterations):
            perturbed_cooler_embeddings = cooler_embeddings + xi * d
            perturbed_out = self.rnn(perturbed_cooler_embeddings)[0]
            perturbed_out = self.head(perturbed_out.mean(1))
            perturbed_logp = F.softmax(perturbed_out, dim=1)

            # Compute KL divergence
            original_out = self.rnn(cooler_embeddings)[0]
            original_out = self.head(original_out.mean(1))
            original_logp = F.softmax(original_out, dim=1)
            kl_div = F.kl_div(perturbed_logp, original_logp, reduction="batchmean")

            # Compute gradients of d using torch.autograd.grad
            grad_d = torch.autograd.grad(kl_div, d, retain_graph=True)[0]
            d = _l2_normalize(grad_d)
            d.requires_grad_()  # Ensure d keeps its gradient requirement

        r_vadv = epsilon * d.detach()  # Detach r_vadv from the computation graph
        # TODO Складывать лосс
        return r_vadv


def _l2_normalize(d):
    d_flat = d.reshape(d.size(0), -1)  # Flatten tensor while keeping the batch dimension
    norm = d_flat.norm(p=2, dim=1, keepdim=True).detach()  # Compute the L2 norm
    d_norm = d_flat / norm  # Normalize
    return d_norm.view_as(d)  # Reshape back to the original dimensions
