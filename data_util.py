"""
Clean the data, extract, store into files
"""
import re
import json
import nltk
import pickle
import numpy as np
from nltk.tree import Tree
from os.path import join as pjoin
from tensorflow.python.platform import gfile

import sys
import codecs

reload(sys)
sys.setdefaultencoding('utf8')

_PAD = b"<pad>"
_SOS = b"<sos>"
_EOS = b"<eos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def data_from_json(filename):
    data = []
    with open(filename) as data_file:
        for l in data_file:
            data.append(json.loads(l))
    return data


def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
    return map(lambda x: x.encode('utf8'), tokens)


def write_to_file(q, root, file_name, text_file=True, pkl_file=True):
    if pkl_file:
        file_path = pjoin(root, file_name + ".pkl")

        with open(file_path, mode="wb") as f:
            pickle.dump(q, f)

    if text_file:
        file_path = pjoin(root, file_name + ".txt")
        with open(file_path, mode="wb") as f:
            for pair in q:
                f.write(str(pair) + b"\n")


# so we can dump in trimmed_q or final_q
# because it's a list of list
def create_dataset(q, root, prefix="", splits=(0.9, 0.05, 0.05), pkl_file=False, text_file=False):
    train_split, val_split, test_split = splits
    total = len(q)

    train_q = q[0:int(np.ceil(total * train_split))]
    val_q = q[int(np.ceil(total * train_split)):int(np.ceil(total * (train_split + val_split)))]
    test_q = q[int(np.ceil(total * (train_split + val_split))):]

    write_to_file(train_q, root, file_name=prefix + "_train", pkl_file=pkl_file, text_file=text_file)
    write_to_file(val_q, root, file_name=prefix + "_val", pkl_file=pkl_file, text_file=text_file)
    write_to_file(test_q, root, file_name=prefix + "_test", pkl_file=pkl_file, text_file=text_file)


def traverseTree(tree, parts):
    # this is pre-order traversal
    # Kelvin's paper uses in-order traversal
    parts.append(tree.label())
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            traverseTree(subtree, parts)
        else:
            parts.append(subtree)

def postprocess(s_list):
    w_list = []
    for c in s_list:
        r_par = c.count(')')
        l_par = c.count('(')
        if r_par > 1:
            c = re.sub('\)\)+', ')', c)
            w_list.append(c)
            for _ in range(r_par - 1):
                w_list.append(')')
        elif l_par > 1:
            c = re.sub('\(\(+', '(', c)
            for _ in range(r_par - 1):
                w_list.append('(')
            w_list.append(c)
        else:
            w_list.append(c)

    return w_list

def src_tokenizer(s):
    words = basic_tokenizer(unicode(s))
    new_words = []
    for w in words:
        r_punct = re.findall(r"[\w']+|[.,!?;\(\)\[\]]", w)

        segs = []
        for seg in r_punct:
            r_multi2 = re.findall(r"[\d+](x)[\d+]", seg)
            r_multi3 = re.findall(r"(\d+)(x)(\d+)(x)(\d+)", seg)
            print(r_multi2)
            print(r_multi3)
            if len(r_multi2) != 0:
                segs.extend(list(r_multi2[0]))
            elif len(r_multi3) != 0:
                segs.extend(list(r_multi3[0]))
            else:
                segs.append(seg)

        new_words.extend(segs)

    return new_words

def linearize(q):
    for line in q:
        parse = line[1]  # this is at least one of the parses
        parsed_tree = Tree.fromstring(parse.replace(":", ""))  # remove all ":"
        sent = line[0]
        # 'repeat 4 [ add red ; select left ] ; select right'

        # Tokenizing everything
        sent = sent.replace("[", " [ ").replace("]", " ] ").replace(";", " ; ")  # so we can tokenize by space
        sent = sent.replace("{", " { ").replace("}", " }").replace("/", " / ")
        sent = sent.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").replace("x", " x ")

        # remove "\\"
        sent = sent.replace("\\", "")

        # more than one blank space becomes just one
        sent = re.sub(' +', ' ', sent)

        parts = []
        traverseTree(parsed_tree, parts)
        str_tree = " ".join(parts)

        line[0] = sent
        line[1] = str_tree

def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def lists_to_indices(list_s, src_vocab, tgt_vocab):
    counter = 0
    tokenized_s = []
    for s in list_s:
        counter += 1
        if counter % 1000 == 0:
            print("tokenizing line %d" % counter)

        src_tokens = src_tokenizer(s[0])
        tgt_tokens = postprocess(basic_tokenizer(s[1]))

        src_token_ids = [src_vocab.get(w, UNK_ID) for w in src_tokens]
        tgt_token_ids = [tgt_vocab.get(w, UNK_ID) for w in tgt_tokens]

        tokenized_s.append([src_token_ids, tgt_token_ids])

    return tokenized_s

def create_vocabulary(list_s, root):
    print("Creating vocabulary %s from data")
    src_vocab = {}
    tgt_vocab = {}

    counter = 0

    for s in list_s:
        counter += 1
        if counter % 1000 == 0:
            print("processing line %d" % counter)

        # tokenize source
        src_tokens = src_tokenizer(s[0])  # hope this is enough

        # tokenize target
        tgt_tokens = postprocess(basic_tokenizer(s[1]))

        for w in src_tokens:
            if w in src_vocab:
                src_vocab[w] += 1
            else:
                src_vocab[w] = 1

        for w in tgt_tokens:
            if w in tgt_vocab:
                tgt_vocab[w] += 1
            else:
                tgt_vocab[w] = 1

    src_vocab_list = _START_VOCAB + sorted(src_vocab, key=src_vocab.get, reverse=True)
    tgt_vocab_list = _START_VOCAB + sorted(tgt_vocab, key=tgt_vocab.get, reverse=True)
    print("Source vocabulary size: %d" % len(src_vocab_list))
    print("Target vocabulary size: %d" % len(tgt_vocab_list))

    with codecs.open(pjoin(root, "src_vocab.dat"), "w", encoding="utf-8") as vocab_file:
        for w in src_vocab_list:
            vocab_file.write(w + "\n")  # .encode('utf-8')
    with codecs.open(pjoin(root, "tgt_vocab.dat"), "w", encoding="utf-8") as vocab_file:
        for w in tgt_vocab_list:
            vocab_file.write(w + "\n")  # .encode('utf-8')

if __name__ == '__main__':
    np.random.seed(123)

    ACC_RATIO = 15  # defines character-wise target/source ratio

    train_file_name = pjoin("queries", "freebuild.json")
    data = data_from_json(train_file_name)
    accept_q = filter(lambda d: "accept" in d['q'], data)
    accept_q = map(lambda d: d['q'], accept_q)

    accept_q = map(lambda s: re.findall('"([^"]*)"', s), accept_q)
    # [[u'add red', u'(: add red here)'], ...]

    len_query = map(lambda s: len(s[1]), accept_q)
    len_inp = map(lambda s: len(s[0]), accept_q)

    ratio = np.array(len_query, dtype=np.float32) / np.array(len_inp, dtype=np.float32)

    accept_idx, = np.nonzero(ratio < ACC_RATIO)

    sat_q = np.take(accept_q, accept_idx).tolist()

    num_parsed_query = map(lambda e: len(e), sat_q)

    select_idx, = np.nonzero(np.array(num_parsed_query) == 2)

    # only select q_s that don't have multiple parses
    small_q = np.take(sat_q, select_idx).tolist()  # 14051

    trimmed_q = map(lambda e: [e[0], e[1]], sat_q)  # 31994

    # ==== We tokenized Small_q ====

    # linearize(small_q)
    #
    # # Vocabulary size: 244
    # create_vocabulary(small_q, "data")
    #
    # # shuffle q
    # np.random.shuffle(small_q)
    #
    # # map inputs to indices...
    # tokenize_q = q_to_token_ids(small_q, pjoin("data", "vocab.dat"))
    #
    # create_dataset(small_q, "data", prefix="small_q", text_file=True)
    # create_dataset(tokenize_q, "data", prefix="tok_small_q", pkl_file=True)

    # ==== We tokenize trimmed_q ===

    # Vocabulary size: 244
    create_vocabulary(trimmed_q, pjoin("data", "nat"))

    src_vocab, rev_src_vocab = initialize_vocabulary(pjoin("data", "nat", "src_vocab.dat"))
    tgt_vocab, rev_tgt_vocab = initialize_vocabulary(pjoin("data", "nat", "tgt_vocab.dat"))

    # shuffle q
    np.random.shuffle(trimmed_q)

    # map inputs to indices...
    tokenized_q = lists_to_indices(trimmed_q, src_vocab, tgt_vocab)

    create_dataset(trimmed_q, "data", prefix="trimmed_q", text_file=True)
    create_dataset(trimmed_q, "data", prefix="trimmed_q", pkl_file=True)