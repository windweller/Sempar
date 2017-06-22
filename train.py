import tensorflow as tf

from rnn import Encoder, Decoder, Seq2SeqNeuralParser
from os.path import join as pjoin
from nltk.tokenize.moses import MosesDetokenizer
from data_util import initialize_vocabulary

import logging
import pickle
import sys
import os

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.5, "Learning rate decays by this much.")
tf.app.flags.DEFINE_integer("learning_rate_decay_epoch", 4, "Learning rate starts decaying after this epoch.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 40, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 5, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("curr_epoch", 0, "Start at epoch n.")
tf.app.flags.DEFINE_integer("size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("input_len", 15, "How much input do we want to keep")
tf.app.flags.DEFINE_integer("query_len", 40, "How much context do we want to keep")  # 400
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_string("dataset", "shrdlurn", "shrdlurn / nat")
tf.app.flags.DEFINE_integer("print_every", 100, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_integer("evaluate", 0, "No training, just evaluation")

FLAGS = tf.app.flags.FLAGS


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def initialize_model(session, model):
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def main(_):
    dataset = FLAGS.dataset

    src_vocab, rev_src_vocab = initialize_vocab(pjoin("data", dataset, "src_vocab.dat"))
    tgt_vocab, rev_tgt_vocab = initialize_vocab(pjoin("data", dataset, "tgt_vocab.dat"))

    if dataset == "shrdlurn":
        pkl_train_name = pjoin("data", dataset, "tokenized_s_train.pkl")
        pkl_val_name = pjoin("data", dataset, "tokenized_s_val.pkl")
    elif dataset == 'nat':
        pkl_train_name = pjoin("data", dataset, "trimmed_q_train.pkl")
        pkl_val_name = pjoin("data", dataset, "trimmed_q_val.pkl")


    with open(pkl_train_name, "rb") as f:
        q_train = pickle.load(f)

    with open(pkl_val_name, "rb") as f:
        q_valid = pickle.load(f)

    encoder = Encoder(size=FLAGS.size)
    decoder = Decoder(size=FLAGS.size, num_layers=1)

    parser = Seq2SeqNeuralParser(encoder, decoder, FLAGS.max_gradient_norm, FLAGS.learning_rate,
                                 FLAGS.learning_rate_decay_factor, FLAGS.dropout, FLAGS.input_len, src_vocab, tgt_vocab,
                                 FLAGS)

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS.train_dir))
    logging.getLogger().addHandler(file_handler)

    with tf.Session() as sess:
        initialize_model(sess, parser)

        if FLAGS.evaluate == 0:
            parser.train(sess, q_train, q_valid, rev_src_vocab, rev_tgt_vocab, FLAGS.epochs, FLAGS.train_dir)


if __name__ == '__main__':
    tf.app.run()
