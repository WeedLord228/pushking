import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('m.model')

sp.vocab_size()

