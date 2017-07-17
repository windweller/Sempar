"""
Clean the data, extract, store into files
also segment SHRDLURN data
"""

import re
import sys
import json
import nltk
import pickle
import codecs
import numpy as np
from nltk.tree import Tree
from os.path import join as pjoin
from tensorflow.python.platform import gfile

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


def data_from_lisp(filename):
    data = []
    with open(filename) as data_file:
        lisp_piece = []
        for line in data_file:
            if not line.startswith(')'):
                lisp_piece.append(line)
            else:
                lisp_piece.append(line)
                data.append(" ".join(lisp_piece))
                lisp_piece = []
    return data


def parse_list(list_str):
    m = re.findall(r"\[([0-9_,]+)\]", list_str)
    m = [[int(i) for i in n.split(",")] for n in m]
    return m


def parse_utterance(s):
    m = re.findall(r'\"(.+?)\"', s)
    if len(m) == 0:
        return False
    return m[0]


# to replace parse_list
def array_to_string(list_str):
    # only for context and TargetValue
    # strip beginning and end '[' and ']'
    # seperate '[' '3' ']'
    m = re.findall(r"\[([0-9_,]+)\]", list_str)
    str_m = []
    for pair in m:
        str_m.append('[ ' + " ".join(pair.split(",")) + ' ] ')
    str_m = "".join(str_m).strip()
    return str_m


def lisp_to_dict(lisps):
    list_dict = []

    for lisp in lisps:
        formula = []
        cont = False
        for line in lisp.split("\n"):
            if "session:" in line:
                session_id = line.split("session:")[1][:-1]
            if "NaiveKnowledgeGraph" in line:
                context = parse_list(line)
            if "utterance" in line:
                utterance = parse_utterance(line)
            if "targetValue" in line:
                target_value = parse_list(line)

            if "targetFormula" in line:
                formula.append(line)
                cont = True
            elif "targetValue" in line:
                cont = False
            elif cont:
                formula.append(line)

        if not utterance:
            continue

        formula = '\n'.join(formula)
        list_dict.append([session_id, context, utterance, formula, target_value])

    return list_dict


# allow ending word to contain a ) parenthese
# usage: postprocess(basic_tokenizer(s))
def postprocess(s_list):
    w_list = []
    for c in s_list:
        r_par = c.count(')')
        if r_par > 1:
            c = re.sub('\)\)+', ')', c)
            w_list.append(c)
            for _ in range(r_par - 1):
                w_list.append(')')
        else:
            w_list.append(c)

    return w_list

def strip_program_prefix(list_s):
    new_list = []
    for s in list_s:
        new_list.append(s.replace("edu.stanford.nlp.sempre.cubeworld.StacksWorld.", ""))
    return new_list

# tokenizer
def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def src_tokenizer(s):
    words = basic_tokenizer(unicode(s))
    new_words = []
    for w in words:
        r = re.findall(r"[\w']+|[.,!?;\(\)]", w)
        if len(r) == 0:  # foreign words
            new_words.append(w)
        else:
            new_words.extend(r)
    return new_words


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
        src_tokens = src_tokenizer(s[2])  # hope this is enough

        # tokenize target
        tgt_tokens = postprocess(basic_tokenizer(s[3]))

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

        src_tokens = src_tokenizer(s[2])
        tgt_tokens = postprocess(basic_tokenizer(s[3]))

        src_token_ids = [src_vocab.get(w, UNK_ID) for w in src_tokens]
        tgt_token_ids = [tgt_vocab.get(w, UNK_ID) for w in tgt_tokens]

        tokenized_s.append([src_token_ids, tgt_token_ids])

    return tokenized_s


# 3. split into train, val, test, save as pickle files
# source/target has cycles (since they are based on the same tutorial/session steps)
# need to randomize

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


def create_dataset(q, root, prefix="", splits=(0.9, 0.05, 0.05), pkl_file=False, text_file=False):
    train_split, val_split, test_split = splits
    total = len(q)

    train_q = q[0:int(np.ceil(total * train_split))]
    val_q = q[int(np.ceil(total * train_split)):int(np.ceil(total * (train_split + val_split)))]
    test_q = q[int(np.ceil(total * (train_split + val_split))):]

    write_to_file(train_q, root, file_name=prefix + "_train", pkl_file=pkl_file, text_file=text_file)
    write_to_file(val_q, root, file_name=prefix + "_val", pkl_file=pkl_file, text_file=text_file)
    write_to_file(test_q, root, file_name=prefix + "_test", pkl_file=pkl_file, text_file=text_file)


if __name__ == '__main__':
    trees = data_from_lisp(pjoin("data", "shrdlurn", "all.lisp"))

    lists = lisp_to_dict(trees)

    create_vocabulary(lists, pjoin("data", "shrdlurn"))

    src_vocab, rev_src_vocab = initialize_vocabulary(pjoin("data", "shrdlurn", "src_vocab.dat"))
    tgt_vocab, rev_tgt_vocab = initialize_vocabulary(pjoin("data", "shrdlurn", "tgt_vocab.dat"))

    tokenized_s = lists_to_indices(lists, src_vocab, tgt_vocab)
    np.random.shuffle(tokenized_s)
    create_dataset(tokenized_s, pjoin("data", "shrdlurn"), prefix="tokenized_s",
                   text_file=True, pkl_file=True)
