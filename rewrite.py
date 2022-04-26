import json
from numpy.random import choice
import tqdm
from pathos.multiprocessing import ProcessPool
import argparse

def norm(original, rules):
    s = sum([c for _,c in rules.items()])
    rules = {a: c/s for a, c in rules.items()}
    if original not in rules:
        rules[original] = 0.0 if s > 0 else 1
    return rules

def load_rules(rules_path, data, goal_wer, max_iter, stop_iter_treshold):
    print('Loading rules...')
    rules = {original: norm(original, forms) for line in open(rules_path).readlines() for original, forms in json.loads(line).items() if not any(map(str.isspace, list(original)))}

    for line in data:
        for word in line:
            if word not in rules.keys():
                rules[word] = {word:1}
    form_lens = {word:{form: sum(w != word for w in form.split()) for form in rules[word].keys()} for word in rules.keys()}
    for word in rules.keys():
        form_lens[word][''] = 1

    GOAL = 1 - goal_wer
    def normalize(rules):
        correction_coef = 1
        total_words = sum(len(line) for line in data)
        E_word = {word: sum([rules[word][form] * form_lens[word][form] for form in rules[word].keys() if form != word]) for word in rules.keys()}
        E_dataset = 1 - sum(E_word[word] for line in data for word in line) / total_words

        correction_coef = GOAL / E_dataset

        def correction(original, rules):
            g = rules[original]
            if g < 1:
                rules[original] = max(min(g * correction_coef, 1), 0)

                b = 1 - g
                nb = 1 - rules[original]
                if b > 0:
                    for k, v in rules.items():
                        if k != original:
                            rules[k] = max(min(v * nb / b, 1), 0)
            return rules
        return {original:correction(original,forms) for original, forms in rules.items()}, E_dataset

    print('Computing substitution probs to match desired WER...')
    E_dataset = 1
    iters = 1
    while abs(E_dataset - GOAL) > stop_iter_treshold and iters <= max_iter:
        rules, E_dataset = normalize(rules)
        print(f'Iter {iters}: Expected WER: {1-E_dataset}')
        iters += 1
    print(f'Finished, expected WER: {1-E_dataset}')
    return rules

def rewrite(r, line):
    o = ''
    for w in line:
        o += choice(list(r[w].keys()), 1, p=list(r[w].values()))[0]
        o += ' '
    return o.strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rules', type=str, help='rules in JSON format obtained using get_rules.py')
    parser.add_argument('goal_wer', type=float, help='goal WER in range ]0;1[')
    parser.add_argument('data', type=str, help='input data in plain text format')
    parser.add_argument('output', type=str, help='output destination')
    parser.add_argument('--max-iter', type=int, help='max iterations of for matching goal WER', default=10)
    parser.add_argument('--stop-iter-treshold', type=int, help='min difference of expected and goal WER in range ]0;1[', default=1e-4)

    args = parser.parse_args()

    assert args.goal_wer < 1 and args.goal_wer > 0

    print('Loading data...')
    if args.data.endswith('json'):
        data = [json.loads(d)['multisource_asr']['normalized'].strip().split() for d in tqdm.tqdm(open(args.data).readlines())]
    else:
        data = [d.strip().split() for d in tqdm.tqdm(open(args.data).readlines())]
    
    rules = load_rules(args.rules, data, args.goal_wer, args.max_iter, args.stop_iter_treshold)
    pool = ProcessPool()

    print('Writing output...')
    with open(args.output, 'w') as output:
        for line in tqdm.tqdm(pool.imap(lambda l: rewrite(rules, l), data), total=len(data)):
            output.write(line)
            output.write('\n')