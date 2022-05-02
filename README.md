# asr-errors-simulator
ASR Errors Simulator

Install:

```
pip install fast-mosestokenizer pathos tqdm Bio numpy
```


Use for rewriting rules:

```
(p3) $ python3 rewrite2.py -h
usage: rewrite2.py [-h] [--punct PUNCT] [--casing CASING] [--cap-start]
                   [--full-stop] [--lang LANG] [--seed [SEED]]
                   rules goal_wer data output

positional arguments:
  rules
  goal_wer         Goal WER, a float between 0.0 and 1.0.
  data             Input file. If -, use stdin.
  output

optional arguments:
  -h, --help       show this help message and exit
  --punct PUNCT    Punctuation option: keep random no. Keep: the punctuation
                   is kept from the input sentence. Random: punctuation tokens
                   are treated as normal tokens, e.g. as OOV if not in rules,
                   and randomly transmitted/substituted/inserted/deleted. No:
                   all punctuation is retrieved from input. It can appear in
                   the output only if it is a part of non-punct-only token, or
                   generated by a rule.
  --casing CASING  Output casing option: keep lower. Keep: keep casing from
                   the input in transmission and substitution. Lower: make
                   everything lowercase.Note that it is assumed that the rules
                   are all in lowercase, so each token is lowercased before
                   searching the rule for it.
  --cap-start      Capitalize the first character of each output line.
  --full-stop      Full-stop: make sure that every line of input is terminated
                   by one punctuation mark, "." by default, or "!", "?" or
                   other if keep or random generates it.
  --lang LANG      Language option for MosesTokenizer. Default is "en".
  --seed [SEED]    Random seed. If this option is unused, the seed is not set.
                   No argument: the seed is 1234.
```
