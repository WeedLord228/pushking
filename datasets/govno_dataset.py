from torch.utils.data import Dataset


class GovnoDataset(Dataset):
    def __init__(self, data_path, sp_processor):
        super().__init__()

        with open(data_path, encoding="UTF-8") as f:
            self.lines = f.readlines()

        self.sp_processor = sp_processor
        self.unk_id = sp_processor.unk_id()
        self.bos_id = sp_processor.bos_id()
        self.eos_id = sp_processor.eos_id()
        self.pad_id = sp_processor.pad_id()

    def __getitem__(self, index):
        pre_result = self.lines[index].split("\t")

        label = pre_result[-1]
        to_process = "".join(pre_result[:-1])

        result = []
        result.extend(self.sp_processor.encode_as_ids(to_process))
        result.append(self.eos_id)

        return result, int(label)

    def __len__(self):
        return len(self.lines)
