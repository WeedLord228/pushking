from utils import replace_in_all_lines

with open("materials/more_data.txt") as file:
    lines = file.readlines()

raw_lines = [x for x in lines if x.startswith("\t\t")]
replacements = [
    (r"&#....", ""),
    (r"\xa0", " "),
    ("…", "..."),
    ("–", "-"),
    ("—", "-"),
    ("-", "-"),
    ("«", '"'),
    ("»", '"'),
    ("„", '"')
]

data = raw_lines
for replacement in replacements:
    data = replace_in_all_lines(replacement[0], replacement[1], data)

data = [x[2:] for x in data]

with open('materials/sentences_nlfi_raw.txt', "w", encoding="UTF-8") as file:
    for line in data:
        file.write(line)

train_threshold = round((len(data) / 100) * 80)

# nlfi - number of lines, french included
TRAIN_FILE_NAME = "materials/train_nlfi_raw.txt"
EVAL_FILE_NAME = "materials/eval_nlfi_raw.txt"

with open(TRAIN_FILE_NAME, "w", encoding="UTF-8") as file:
    for line in data[:train_threshold]:
        file.write(line)

with open(EVAL_FILE_NAME, "w", encoding="UTF-8") as file:
    for line in data[train_threshold:]:
        file.write(line)

print(f"Clean, processed data stored in '{TRAIN_FILE_NAME}' and '{EVAL_FILE_NAME}'")
