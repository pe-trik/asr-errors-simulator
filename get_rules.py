import argparse
import json
import tqdm
from Bio import pairwise2
from pathos.multiprocessing import ProcessingPool
import editdistance

def main(args):
    src_sentences, tgt_sentences = [], []
    rewrite_rules = {}

    for line in tqdm.tqdm(open(args.corpus, 'r').readlines()):
        src, tgt = line.strip().split(args.delimiter)
        src_sentences.append(src)
        tgt_sentences.append(tgt)

    def add_pair(s, t):
        if s not in rewrite_rules.keys():
            rewrite_rules[s] = dict([(t,1)])
        else:
            rules = rewrite_rules[s]
            if t not in rules.keys():
                rules[t] = 1
            else:
                rules[t] += 1
    
    def get_rules(a, b):
        head, tail = '', ''
        for s, t in zip(a, b):
            if s == ' ':#(s == ' ' and t == ' ')  or (s == '-' and t == ' ') or (s == ' ' and t == '-'):
                if len(head) > 0:
                    stail = tail.split()
                    if len(stail) > 1:
                        dists = [editdistance.distance(list(head), list(t)) for t in stail]
                        md = min(dists)
                        #print(head, stail, dists)
                        if md > editdistance.distance(list(head), list(tail.replace(' ', ''))):
                            #print('\t',head, tail)
                            yield (head, tail)
                        else:
                            for d, t in zip(dists, stail):
                                #print('\t', ('' if d > md else head, t))
                                yield ('' if d > md else head, t)
                    else:
                        yield (head, tail)
                head, tail = '', ''
            else:
                if s != '-':
                    head += s
                if t != '-':
                    tail += t
        if len(head) > 0:
            yield (head, tail)

    def norm(rules):
        s = sum([c for _,c in rules.items()])
        rules = {a: c/s for a, c in rules.items()}
        return rules

    pool = ProcessingPool()
    if args.chacter_level_alignment:
        def func(args):
            return pairwise2.align.globalms(list(args[0]), list(args[1]), 0, -1, -1, -1, one_alignment_only=True, gap_char=['-'])[0]
    else:
        def match_fn(a, b):
            if min(len(a),len(b)) == 0:
                return 0
            ml = max(len(a),len(b))
            return 1 - editdistance.eval(a, b) / ml
        def func(args):
            return pairwise2.align.globalcx(args[0].split(), args[1].split(), one_alignment_only=True, gap_char=['-'], match_fn=match_fn)[0]
    alignments = pool.imap(func, zip(src_sentences, tgt_sentences))
    
    pairs = []
    break_rules = get_rules if args.chacter_level_alignment else zip
    for alignment in tqdm.tqdm(alignments, total=len(src_sentences)):
        for s, t in break_rules(alignment.seqA, alignment.seqB):
            pairs.append((s, t))
    for s,t in tqdm.tqdm(pairs):
        add_pair(s,t)
    
    filtered_rewrite_rules = dict()
    for orignal, recognized in rewrite_rules.items():
        orignal = orignal.replace('-', '')
        filtered = filtered_rewrite_rules[orignal] if orignal in filtered_rewrite_rules.keys() else dict()

        for r, c in recognized.items():
            r = r.replace('-', '')
            if orignal != '' and r != '' and editdistance.eval(list(r), list(orignal)) > min(len(r), len(orignal)):
                continue
            if r in filtered.keys():
                filtered[r] += c
            else:
                filtered[r] = c

        if '' not in filtered:
            filtered[''] = 0

        if orignal not in filtered:
            filtered[orignal] = 0
        
        filtered_rewrite_rules[orignal] = filtered
    rewrite_rules = filtered_rewrite_rules

    insertions, deletions, substitutions, no_error, total = 0,0,0,0,0
    for orignal, recognized in rewrite_rules.items():
        deletions += recognized['']
        del recognized['']   
        total_rule = sum(v for v in recognized.values())
        if orignal != '':
            substitutions += total_rule - recognized[orignal]
            no_error += recognized[orignal]
            del recognized[orignal]
        else:
            insertions += total_rule
        
        total += total_rule
    total += deletions

    with open(args.rules_output, 'w') as file:
        file.write(json.dumps(
                {'<unk>': {'p_insert':insertions / total, 'p_delete':deletions / total, 'p_transmit':(substitutions + no_error) / total, 'p_substitute':substitutions / (substitutions + no_error)}}, ensure_ascii=False
            ))
        file.write('\n')
        for orignal, recognized in tqdm.tqdm(rewrite_rules.items()):
            recognized = norm(recognized)
            file.write(json.dumps(
                {orignal: recognized}, ensure_ascii=False
            ))
            file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', type=str)
    parser.add_argument('rules_output', type=str)
    parser.add_argument('--delimiter', type=str, default='|')
    parser.add_argument('--chacter_level_alignment', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
