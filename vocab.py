import pickle
import argparse
import os
from collections import Counter
import codecs

def path_to_vocab():
    return os.path.join('', 'vocab.pkl')


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
        
    def __len__(self):
        return len(self.word2idx)

    def start_token(self):
        return '<start>'

    def end_token(self):
        return '<end>'


def build_vocab(tsv='covost_v2.fr_en.train.tsv', field_num=2):
    print("TSV : %s" % tsv)

    f = codecs.open(tsv, encoding='utf-8')
    field_num = field_num
    lines = f.readlines()
    counter = Counter()

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word(vocab.start_token())
    vocab.add_word(vocab.end_token())
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for line in lines:
        txt = line.lower().split('\t')[field_num]
        for char in txt:
            counter.update(char)

    words = counter.most_common(70)
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in words if cnt >= 10]

    for i, word in enumerate(words):
        vocab.add_word(word)
    

    print('Total number of words in vocab:', len(vocab))

    return vocab

def dump_vocab(path=path_to_vocab(), tsv='covost_v2.fr_en.train.tsv', field_num=2):
    if not os.path.exists(path):
        vocab = build_vocab(tsv, field_num)
        with open(path, 'wb') as f:
            pickle.dump(vocab, f)
        print("Total vocabulary size: %d" %len(vocab))
        print("Saved the vocabulary wrapper to '%s'" %path)
    else:
        print('Vocabulary already exists.')

def load_vocab(path=path_to_vocab()):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError('Failed to load %s: %s' % (path, e))


def main(args):
    vocab_path = args.vocab_path
    dump_vocab(vocab_path, args.tsv_path, args.field_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_path', type=str, 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default=path_to_vocab(),
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--field_num', type=int, default=2, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
