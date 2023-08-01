from utils import replace_in_all_lines

with open("materials/more_data.txt") as file:
    lines = file.readlines()

raw_lines = [x for x in lines if x.startswith("\t\t")]
no_dia_lines = replace_in_all_lines(r"&#....", "", raw_lines)
data = replace_in_all_lines(r"\xa0", " ", no_dia_lines)
data = [x[2:] for x in data]

train_threshold = round((len(data) / 100) * 80)

# nlfi - number of lines, french included
TRAIN_FILE_NAME = "materials/train_nlfi.txt"
EVAL_FILE_NAME = "materials/eval_nlfi.txt"

with open(TRAIN_FILE_NAME, "w", encoding="UTF-8") as file:
    for line in data[:train_threshold]:
        file.write(line)

with open(EVAL_FILE_NAME, "w", encoding="UTF-8") as file:
    for line in data[train_threshold:]:
        file.write(line)

print(f"Clean, processed data stored in '{TRAIN_FILE_NAME}' and '{EVAL_FILE_NAME}'")
