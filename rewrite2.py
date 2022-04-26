import json
from numpy.random import choice, random_sample, randint
import tqdm
from pathos.multiprocessing import ProcessPool
import argparse

CORPUS_INFO_WORD = "<CORPUS_INFO_WORD>"

def load_rules(rules_path, goal_wer):
    rules = {original: forms for line in open(rules_path).readlines() for original, forms in json.loads(line).items()}
    unk = rules[CORPUS_INFO_WORD]
    a = (unk['p_insert'] + unk['p_delete']) * 1*unk['p_substitute']
    b = -(unk['p_insert'] + unk['p_delete'] + 1*unk['p_substitute'])
    c = goal_wer
    D = b * b - 4 * a * c
    c1 = (-b + D ** 0.5) / (2 * a)
    c2 = (-b - D ** 0.5) / (2 * a)
    c = min(c1, c2) if min(c1, c2) > 0 else max(c1, c2)
    print('c1:',c1,'c2:', c2, 'c:',c )
    print(unk)
    unk['p_insert'] *= c
    unk['p_delete'] *= c
    unk['p_substitute'] *= 1*c
    unk['p_transmit'] = (1 - unk['p_insert'] - unk['p_delete']) / (1 - unk['p_insert'])
    unk['p_insert'] = unk['p_insert'] / (1 + unk['p_insert'])
    print(unk)
    print('wer:',unk['p_insert'] / (1 - unk['p_insert']) + unk['p_delete'] + (1-unk['p_insert'] - unk['p_delete'])*unk['p_substitute'])
    vocab = set()
    for forms in rules.values():
        vocab.update(forms.keys())
    rules['<vocab>'] = list(vocab)
    return rules




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rules', type=str)
    parser.add_argument('goal_wer', type=float)
    parser.add_argument('data', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()


    data = [line for line in open(args.data).readlines()]
    r = load_rules(args.rules, args.goal_wer)

    unk = r[CORPUS_INFO_WORD]
    vocab = r['<vocab>']
    def rewrite(line):
        o = ''
        for w in line.lower().strip().split():
            while random_sample() < unk['p_insert']:
                o += choice(list(r[''].keys()), 1, p=list(r[''].values()))[0] + ' '

            if random_sample() < unk['p_transmit']:
                if random_sample() < unk['p_substitute']:
                    if w not in r.keys() or len(r[w]) == 0:
                        idx = randint(0, len(vocab))
                        o += vocab[idx] + ' '
                    else:
                        o += choice(list(r[w].keys()), 1, p=list(r[w].values()))[0] + ' '
                else:
                    o += w + ' '

        return o.strip()

    pool = ProcessPool()

    with open(args.output, 'w') as output:
        for line in tqdm.tqdm(pool.imap(lambda l: rewrite(l), data), total=len(data)):
            output.write(line)
            output.write('\n')