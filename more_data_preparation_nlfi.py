from utils import replace_in_all_lines

with open('materials/more_data.txt') as file:
    lines = file.readlines()

raw_lines = [x for x in lines if x.startswith('\t\t')]
no_dia_lines = replace_in_all_lines(r'&#....', '', raw_lines)
data = replace_in_all_lines(r'\xa0', ' ', no_dia_lines)
data = [x[2:] for x in data]

train_threshold = round((len(data) / 100) * 80)

# nlfi - number of lines, french included
train_file_name = 'materials/train_nlfi.txt'
eval_file_name = 'materials/eval_nlfi.txt'
train_file_mode = 'w'
eval_file_mode = 'w'

with open(train_file_name, train_file_mode, encoding='UTF-8') as file:
    for line in data[:train_threshold]:
        file.write(line)

with open(eval_file_name, eval_file_mode, encoding='UTF-8') as file:
    for line in data[train_threshold:]:
        file.write(line)

print(f"Clean, processed data stored in '{train_file_name}' and '{eval_file_name}'")
