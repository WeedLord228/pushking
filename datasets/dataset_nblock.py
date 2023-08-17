from torch.utils.data import Dataset


class DatasetNBlock(Dataset):
    def __init__(self, data, sp_processor, n_tokens):
        self.sp_processor = sp_processor
        self.tokens = []

        if isinstance(data, str):
            with open(data, encoding="UTF-8") as f:
                for line in f.readlines():
                    self.tokens.extend(self.sp_processor.encode_as_ids(line))
        else:
            raise TypeError("Only filename as String is appropriate as data to dataloader")

        self.n_tokens = n_tokens
        self.unk_id = sp_processor.unk_id()
        self.bos_id = sp_processor.bos_id()
        self.eos_id = sp_processor.eos_id()
        self.pad_id = sp_processor.pad_id()

    def __getitem__(self, idx):
        result = self.tokens[idx * self.n_tokens : ((idx + 1) * self.n_tokens) + 1]
        return result[:-1], result[1:]

    def __len__(self):
        return (
            len(self.tokens) // self.n_tokens
            if (len(self.tokens) % self.n_tokens) == 0
            else (len(self.tokens) // self.n_tokens) + 1
        )
