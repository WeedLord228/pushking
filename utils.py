import re

import torch


def replace_in_all_lines(template, replacement, lines):
    result = []
    for line in lines:
        a = re.sub(template, replacement, line)
        if a:
            result.append(a)

    assert len(result) == len(lines)
    return result


def collate_fn_padding_offseted_targets(input_batch, pad_id):
    max_sent_len = max([len(x[0]) for x in input_batch])
    result_x = []
    result_y = []

    for X, Y in input_batch:
        new_x = X
        new_y = Y
        new_x.extend([pad_id] * (max_sent_len - len(X)))
        new_y.extend([pad_id] * (max_sent_len - len(Y)))
        result_x.append(new_x)
        result_y.append(new_y)

    return torch.LongTensor(result_x), torch.LongTensor(result_y)
