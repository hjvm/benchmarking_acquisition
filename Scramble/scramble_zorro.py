import os, sys, random
from rule_based import read_dir, read_zorrofile

indir = sys.argv[1]
outdir = sys.argv[2]

def gen_scrambles_consistent(outdir, fname, pairs, numsamples):
    def scramble(sent, indices):
        return [sent[i] for i in indices if i < len(sent)]
    def write(f, sent):
        f.write(" ".join(sent))
        f.write("\n")

    for seed in range(numsamples):
        random.seed(seed)
        with open(os.path.join(outdir, str(seed), fname), "w") as fout:
            for pair in pairs:
                ungr, gr = pair
                indices = list(range(max(len(ungr), len(gr))))
                random.shuffle(indices)
                ungr_scramble = scramble(ungr, indices)
                gr_scramble = scramble(gr, indices)
                if len(ungr) != len(ungr_scramble):
                    print(ungr, ungr_scramble)
                    exit()
                if len(gr) != len(gr_scramble):
                    print(gr, gr_scramble)
                    exit()
                write(fout, ungr_scramble)
                write(fout, gr_scramble)

if __name__=="__main__":
    fnames = read_dir(indir)

    for fname in fnames:
        print(os.path.basename(fname))
        pairs = read_zorrofile(fname)
        gen_scrambles_consistent(outdir, os.path.basename(fname), pairs, numsamples=10)

