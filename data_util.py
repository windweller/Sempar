"""
Clean the data, extract, store into files
"""
import re
import json
from os.path import join as pjoin
import nltk

def data_from_json(filename):
    with open(filename) as data_file:
        data = json.loads(data_file)
    return data

def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
    return map(lambda x:x.encode('utf8'), tokens)

if __name__ == '__main__':
    train_file_name = pjoin("queries", "freebuild.json")
    data = data_from_json(train_file_name)
    accept_q = filter(lambda d: "accept" in d['q'], data)
    accept_q = map(lambda d: d['q'], accept_q)

    accept_q = map(lambda s: re.findall('"([^"]*)"', s), accept_q)
    # [[u'add red', u'(: add red here)'], ...]

