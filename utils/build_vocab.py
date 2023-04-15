import nltk
import pickle
import argparse
import json
from collections import Counter
from utils.pycocotools.coco import COCO


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

def build_vocab(json_file, threshold):
    """Build a simple vocabulary wrapper."""
    json_data = json.loads(open(json_file, 'r').read())
    counter = Counter()
    for each_dict in json_data:
        reg_list = each_dict['regions']
        for each_region in reg_list:
            caption = each_region['phrase']
            # print(caption)
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--caption_path', type=str,
    #                     default='/usr/share/mscoco/annotations/captions_train2014.json',
    #                     help='path for train annotation file')
    # parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
    #                     help='path for saving vocabulary wrapper')
    # parser.add_argument('--threshold', type=int, default=4,
    #                     help='minimum word count threshold')
    # args = parser.parse_args()
    # main(args)

    # json_data = json.loads(open('vg_data/region_descriptions.json', 'r').read())
    # print(len(json_data))
    # print(json_data[0]['regions'][0]['phrase'])
    # counter = Counter()
    # man_counter = 0
    # for each_dict in json_data:
    #     reg_list = each_dict['regions']
    #     for each_region in reg_list:
    #         caption = each_region['phrase']
    #         print(caption)
    #         tokens = nltk.tokenize.word_tokenize(caption.lower())
    #         counter.update(tokens)
    #         man_counter += 1
    #         if man_counter > 6:
    #             break
    #     if man_counter > 6:
    #         break
    #
    # for word, cnt in counter.items():
    #     print(word, "\t", cnt)
    vocab = build_vocab('vg_data/region_descriptions.json', 10)
    print(len(vocab.word2idx))
    print(vocab.word2idx['<start>'])
    print(vocab.word2idx['<end>'])
    with open('vg_data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" % len(vocab))

