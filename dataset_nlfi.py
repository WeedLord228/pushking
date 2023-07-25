from torch.utils.data import Dataset


class DatasetNlfi(Dataset):
    def __init__(self, data, sp_processor, n_lines):

        if isinstance(data, list):
            self.lines = data
        elif isinstance(data, str):
            with open(data, encoding='UTF-8') as f:
                self.lines = f.readlines()
        else:
            raise TypeError("Only filename as String and lists are appropriate as data to dataloader")

        self.n_lines = n_lines
        self.sp_processor = sp_processor
        self.unk_id = sp_processor.unk_id()
        self.bos_id = sp_processor.bos_id()
        self.eos_id = sp_processor.eos_id()
        self.pad_id = sp_processor.pad_id()

    def __getitem__(self, idx):
        to_tokenize_lines = ''.join(self.lines[idx * self.n_lines:(idx + 1) * self.n_lines])
        result = [self.bos_id]
        result.extend(self.sp_processor.encode_as_ids(to_tokenize_lines))
        result.append(self.eos_id)
        return result[:-1], result[1:]

    def __len__(self):
        return len(self.lines) // self.n_lines if \
            (len(self.lines) % self.n_lines) == 0 else \
            (len(self.lines) // self.n_lines) + 1
