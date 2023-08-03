import re

import torch


def replace_in_all_lines(template, replacement, lines):
    result = []
    for line in lines:
        replaced = re.sub(template, replacement, line)
        if replaced:
            result.append(replaced)

    assert len(result) == len(lines)
    return result


def collate_fn_padding_offseted_targets(input_batch, pad_id):
    max_sent_len = max([len(x[0]) for x in input_batch])  # pylint: disable=R1728
    result_x = []
    result_y = []

    for x, y in input_batch:
        new_x = x
        new_y = y
        new_x.extend([pad_id] * (max_sent_len - len(x)))
        new_y.extend([pad_id] * (max_sent_len - len(y)))
        result_x.append(new_x)
        result_y.append(new_y)

    return torch.LongTensor(result_x), torch.LongTensor(result_y)
