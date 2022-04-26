import argparse
import numpy as np
import numpy.random as npr
import editdistance
import unicodedata

def wer_confidence(hypotheses, references, samples = 1000, confidence = 0.95, cer = False):
    s = []
    for h, r in zip(hypotheses, references):
        h, r = unicodedata.normalize('NFKC', h), unicodedata.normalize('NFKC', r)
        if cer:
            h, r = list(h), list(r)
        else:
            h, r = h.split(), r.split()
        s.append((len(r) ,editdistance.eval(h, r)))
    s = np.array(s)
    wer = np.sum(s[:,1])/np.sum(s[:,0])

    stats = []
    for _ in range(samples):
        idxs = npr.choice(len(s), len(s))
        stats.append(np.sum(s[idxs,1])/np.sum(s[idxs,0]))
    stats = np.sort(stats)
    confidence *= 100
    l = np.percentile(stats, (100 - confidence) / 2.0)
    h = np.percentile(stats, 100 - (100 - confidence) / 2.0)
    return wer * 100, l*100, h*100, stats*100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ref', type=str)
    parser.add_argument('asr', type=str)
    parser.add_argument('--lower', action='store_true', default=False)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--cer', action='store_true',default=False)

    args = parser.parse_args()

    def normalize(txt):
        if args.lower:
            txt = txt.lower()
        return txt.strip()

    refs = [normalize(line) for line in open(args.ref, 'r').readlines()]
    asr = [normalize(line) for line in  open(args.asr, 'r').readlines()]
    w, l, u, _ = wer_confidence(asr, refs, args.samples, args.confidence, args.cer)
    print(f"{'CER' if args.cer else 'WER'} (%):\t{w:.2f}")
    print(f"Bounds (%):\t{l:.2f} - {u:.2f} (with {args.confidence:.3f} confidence)")

