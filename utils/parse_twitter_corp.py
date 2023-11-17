import pandas as pd

val_size = 2000
train_size = 50000

pos_df = pd.read_csv("C:/SeriousStuff/Notebooks/NLP/BONUS.pushking/materials/twitter_corp/positive.csv")
pos_df = pos_df[["ttext", "ttype"]]
neg_df = pd.read_csv("C:/SeriousStuff/Notebooks/NLP/BONUS.pushking/materials/twitter_corp/negative.csv")
neg_df = neg_df[["ttext", "ttype"]]

val_pos_df = pos_df[:val_size]
val_neg_df = neg_df[:val_size]

train_pos_df = pos_df[val_size : train_size + val_size]
train_neg_df = neg_df[val_size : train_size + val_size]

train = pd.concat([train_pos_df, train_neg_df]).sample(frac=1)
train["ttext"] = train["ttext"].replace(to_replace="\r?\n|\r", value=" ", regex=True)
train["ttext"] = train["ttext"].replace(to_replace="\)", value="", regex=True)
train["ttext"] = train["ttext"].replace(to_replace="\(", value="", regex=True)
train["ttype"] = ((train["ttype"] + 1) / 2).astype(int)
val = pd.concat([val_pos_df, val_neg_df]).sample(frac=1)
val["ttext"] = val["ttext"].replace(to_replace="\r?\n|\r", value=" ", regex=True)
val["ttext"] = val["ttext"].replace(to_replace="\)", value="", regex=True)
val["ttext"] = val["ttext"].replace(to_replace="\(", value="", regex=True)
val["ttype"] = ((val["ttype"] + 1) / 2).astype(int)

val.to_csv(
    header=False,
    index=False,
    sep="\t",
    path_or_buf="C:/SeriousStuff/Notebooks/NLP/BONUS.pushking/materials/twitter_corp/val_no_par.csv",
)

train.to_csv(
    header=False,
    index=False,
    sep="\t",
    path_or_buf="C:/SeriousStuff/Notebooks/NLP/BONUS.pushking/materials/twitter_corp/train_no_par.csv",
)
