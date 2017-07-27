# Copyright 2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# modified from train_nlc.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import random
import json
import pickle

import numpy as np
from six.moves import xrange
import tensorflow as tf
from os.path import join as pjoin

import rnn_logic
from data_util import initialize_vocabulary
import data_util

from util import exact_match_score, f1_score

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.0003, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 3, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")  # was 250
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")  # originally 2 layers
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "sandbox", "Training directory.")
tf.app.flags.DEFINE_string("tokenizer", "BPE", "BPE / CHAR / WORD.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 20, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("dataset", "shrdlurn", "shrdlurn / nat")
tf.app.flags.DEFINE_integer("input_len", 15, "How much input do we want to keep")
tf.app.flags.DEFINE_integer("query_len", 35, "How much query do we want to keep")
tf.app.flags.DEFINE_integer("beam_size", 3, "Size of beam.")
tf.app.flags.DEFINE_boolean("print_decode", False, "print decoding result to file. Is slow.")
tf.app.flags.DEFINE_boolean("co_attn", False, "Whether to use co-attention to encode")
tf.app.flags.DEFINE_boolean("seq", False, "Whether to sequence encode")
tf.app.flags.DEFINE_boolean("cat_attn", False, "Whether to use concatenated representation with decoder attention")
tf.app.flags.DEFINE_integer("seed", 123, "random seed to use")
tf.app.flags.DEFINE_boolean("dev", False, "Skip training and generate output files to eval folder")
tf.app.flags.DEFINE_integer("best_epoch", 0, "Specify the best epoch to use")
# we want this for precise control, valid cost is not what we want
tf.app.flags.DEFINE_string("restore_checkpoint", None, "checkpoint file to restore model parameters from, without ckpt suffix")

FLAGS = tf.app.flags.FLAGS

_PAD = b"<pad>"
_SOS = b"<sos>"
_EOS = b"<eos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


# load in old model by giving the same train_dir
def create_model(session, src_vocab_size, tgt_vocab_size, env_vocab_size, forward_only):
    model = rnn_logic.NLCModel(
        src_vocab_size, tgt_vocab_size, env_vocab_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.dropout, FLAGS,
        forward_only=forward_only, optimizer=FLAGS.optimizer)
    # so now we can load safely
    # ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    # v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    # if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
    #     logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    #     model.saver.restore(session, ckpt.model_checkpoint_path)
    # else:
    logging.info("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
    logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def validate(model, sess, q_valid):
    valid_costs, valid_lengths = [], []
    for source_tokens, source_mask, target_tokens, target_mask, \
        ctx_tokens, ctx_mask, pred_tokens, pred_mask in pair_iter(q_valid, FLAGS.batch_size,
                                                                            FLAGS.input_len, FLAGS.query_len):
        cost = model.test_engine(sess, source_tokens.T, source_mask.T, ctx_tokens.T, ctx_mask.T,
                                 pred_tokens.T, pred_mask.T)
        valid_costs.append(cost * target_mask.shape[1])
        valid_lengths.append(np.sum(target_mask[1:, :]))
    valid_cost = sum(valid_costs) / float(sum(valid_lengths))
    return valid_cost


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


def add_sos_eos(tokens):
    return map(lambda token_list: [data_util.SOS_ID] + token_list + [data_util.EOS_ID], tokens)


def padded(tokens, batch_pad=0):
    maxlen = max(map(lambda x: len(x), tokens)) if batch_pad == 0 else batch_pad
    return map(lambda token_list: token_list + [data_util.PAD_ID] * (maxlen - len(token_list)), tokens)


def pair_iter(q, batch_size, inp_len, query_len):
    # use inp_len, query_len to filter list
    batched_input = []
    batched_query = []
    batched_context = []
    batched_pred = []

    iter_q = q[:]

    while len(iter_q) > 0:
        while len(batched_input) < batch_size and len(iter_q) > 0:
            pair = iter_q.pop(0)
            if len(pair[0]) <= inp_len and len(pair[1]) <= query_len:
                batched_input.append(pair[0])
                batched_query.append(pair[1])
                batched_context.append(pair[2])
                batched_pred.append(pair[3])

        padded_input = np.array(padded(batched_input), dtype=np.int32)
        input_mask = (padded_input != data_util.PAD_ID).astype(np.int32)

        batched_query = add_sos_eos(batched_query)
        padded_query = np.array(padded(batched_query), dtype=np.int32)
        query_mask = (padded_query != data_util.PAD_ID).astype(np.int32)

        padded_ctx = np.array(padded(batched_context), dtype=np.int32)
        ctx_mask = (padded_ctx != data_util.PAD_ID).astype(np.int32)

        batched_pred = add_sos_eos(batched_pred)
        padded_pred = np.array(padded(batched_pred), dtype=np.int32)
        pred_mask = (padded_pred != data_util.PAD_ID).astype(np.int32)

        yield padded_input, input_mask, padded_query, query_mask, \
              padded_ctx, ctx_mask, padded_pred, pred_mask
        batched_input, batched_query, batched_context, batched_pred = [], [], [], []


def decode_beam(model, sess, encoder_output, max_beam_size):
    toks, probs = model.decode_beam(sess, encoder_output, beam_size=max_beam_size)
    return toks.tolist(), probs.tolist()


# what's the diff between this and detokenize_tgt?
def detokenize(sents, reverse_vocab, decode=False):
    space = ""
    if decode:
        space = " "

    def detok_sent(sent):
        outsent = ''
        for t in sent:
            if t >= len(_START_VOCAB):
                outsent += reverse_vocab[t] + space  # this might not be right...
        return outsent

    return [detok_sent(s) for s in sents]


def decode_validate_logic(model, sess, q_valid, reverse_src_vocab,
                          reverse_tgt_vocab, reverse_env_vocab, save_dir, epoch, sample=5, print_decode=False):

    # if print_decode=True, we overwrite FLAGS.print_decode
    # so when flag dev, we don't need to add print_decode
    print_decode = print_decode if print_decode else FLAGS.print_decode
    num_decoded = 0

    # add f1, em measure on this decoding
    f1 = 0.
    em = 0.
    saved_list = []

    # since we did beam-decode, I can measure EM on the top-5 result
    # can split pred / context vocab as well...but it feels like a trick
    with open(pjoin(save_dir, "valid_decode_e" + str(epoch) + ".txt"), "wb") as f:
        for source_tokens, source_mask, target_tokens, target_mask, \
            ctx_tokens, ctx_mask, pred_tokens, pred_mask in pair_iter(q_valid, 1,
                                                                      FLAGS.input_len, FLAGS.query_len):

            # transpose them because how this model is set up
            source_tokens, source_mask, target_tokens, target_mask = source_tokens.T, source_mask.T, target_tokens.T, target_mask.T

            # transpose ctx and pred
            ctx_tokens, ctx_mask, pred_tokens, pred_mask = ctx_tokens.T, ctx_mask.T, pred_tokens.T, pred_mask.T

            # detokenize query and logical form
            # seems like detokenize can handle batch
            src_sent = detokenize(source_tokens, reverse_src_vocab)
            tgt_sent = detokenize(target_tokens, reverse_tgt_vocab)

            # detokenize context and prediction
            ctx_env = detokenize(ctx_tokens, reverse_env_vocab)
            pred_env = detokenize(pred_tokens, reverse_env_vocab)

            # Encode
            encoder_output = model.encode(sess, source_tokens, source_mask, ctx_tokens, ctx_mask)
            # Decode
            beam_toks, probs = decode_beam(model, sess, encoder_output, FLAGS.beam_size)
            # De-tokenize
            beam_strs = detokenize(beam_toks, reverse_tgt_vocab, decode=True)

            best_str = beam_strs[0]  # we can also get probability on them

            num_decoded += 1

            f1 += f1_score(best_str, " ".join(tgt_sent[1:]))
            # tgt_sent's first array element is always [""]
            em += exact_match_score(best_str, " ".join(tgt_sent[1:]))

            if num_decoded <= sample:
                logging.info("cmd: {}".format(" ".join(src_sent)))
                logging.info("ctx: {}".format(" ".join(ctx_env)))
                logging.info("truth: {}".format(" ".join(tgt_sent[1:])))
                logging.info("decoded: {}".format(best_str))
                logging.info("")

            if print_decode:
                f.write("cmd: {} \r".format(" ".join(src_sent)))
                f.write("ctx: {} \r".format(" ".join(ctx_env)))
                f.write("truth: {} \r".format(" ".join(tgt_sent[1:])))
                f.write("decoded: {} \r".format(best_str))
                f.write("\r")
                f.write("\r")
                saved_list.append({"cmd": src_sent,
                                   "ctx": ctx_env,
                                   "truth": tgt_sent[1:],
                                   "decoded": best_str})
    if print_decode:
        with open(pjoin(save_dir, "valid_decode_e" + str(epoch) + ".pkl"), "wb") as f:
            pickle.dump(saved_list, f)

    return float(f1) / float(num_decoded), float(em) / float(num_decoded)


def train():

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS.train_dir))
    logging.getLogger().addHandler(file_handler)

    dataset = FLAGS.dataset

    logging.info("Preparing %s data in %s" % (FLAGS.dataset, FLAGS.data_dir))

    src_vocab, rev_src_vocab = initialize_vocab(pjoin("data", dataset, "src_vocab.dat"))
    tgt_vocab, rev_tgt_vocab = initialize_vocab(pjoin("data", dataset, "tgt_vocab.dat"))
    env_vocab, rev_env_vocab = initialize_vocab(pjoin("data", dataset, "env_vocab.dat"))

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

    logging.info("Source vocabulary size: %d" % len(rev_src_vocab))
    logging.info("Target vocabulary size: %d" % len(rev_tgt_vocab))
    logging.info("Env vocabulary size: %d" % len(rev_env_vocab))

    decode_save_dir = pjoin(FLAGS.train_dir, "eval")
    if not os.path.exists(decode_save_dir):
        os.makedirs(decode_save_dir)

    logging.info(vars(FLAGS))
    with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Graph().as_default(), tf.Session() as sess:
        logging.info("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

        # fix the model loading
        # so if FLAGS.dev, then we don't create ADAM variables (a temporary fix maybe)
        model = create_model(sess, len(rev_src_vocab), len(rev_tgt_vocab), len(rev_env_vocab), FLAGS.dev)

        # manually load best epoch here
        if FLAGS.restore_checkpoint is not None:
            saver = tf.train.import_meta_graph(FLAGS.restore_checkpoint + ".meta")
            saver.restore(sess, FLAGS.restore_checkpoint + ".ckpt")

        if not FLAGS.dev:
            logging.info('Initial validation cost: %f' % validate(model, sess, q_valid))

            epoch = 0
            best_epoch = 0
            previous_losses = []
            previous_ems = []
            exp_cost = None
            exp_length = None
            exp_norm = None
            while (FLAGS.epochs == 0 or epoch < FLAGS.epochs):
                epoch += 1
                current_step = 0

                ## Train
                epoch_tic = time.time()
                for source_tokens, source_mask, target_tokens, target_mask, \
                    ctx_tokens, ctx_mask, pred_tokens, pred_mask in pair_iter(q_train, FLAGS.batch_size,
                                                                                        FLAGS.input_len, FLAGS.query_len):
                    # Get a batch and make a step.
                    tic = time.time()

                    grad_norm, cost, param_norm = model.train_engine(sess, source_tokens.T, source_mask.T, ctx_tokens.T,
                                                                     ctx_mask.T, target_tokens.T, target_mask.T)

                    toc = time.time()
                    iter_time = toc - tic
                    current_step += 1

                    lengths = np.sum(target_mask, axis=0)
                    mean_length = np.mean(lengths)
                    std_length = np.std(lengths)

                    if not exp_cost:
                        exp_cost = cost
                        exp_length = mean_length
                        exp_norm = grad_norm
                    else:
                        exp_cost = 0.99 * exp_cost + 0.01 * cost
                        exp_length = 0.99 * exp_length + 0.01 * mean_length
                        exp_norm = 0.99 * exp_norm + 0.01 * grad_norm

                    cost = cost / mean_length

                    if current_step % FLAGS.print_every == 0:
                        logging.info(
                            'epoch %d, iter %d, cost %f, exp_cost %f, grad norm %f, param norm %f, batch time %f, length mean/std %f/%f' %
                            (epoch, current_step, cost, exp_cost / exp_length, grad_norm, param_norm, iter_time,
                             mean_length,
                             std_length))
                epoch_toc = time.time()

                ## Checkpoint
                checkpoint_path = os.path.join(FLAGS.train_dir, "best.ckpt")

                ## Validate
                valid_cost = validate(model, sess, q_valid)

                # Validate by decoding
                f1, em = decode_validate_logic(model, sess, q_valid, rev_src_vocab, rev_tgt_vocab, rev_env_vocab,
                                               decode_save_dir, epoch, sample=5)

                logging.info("Epoch %d Validation cost: %f time: %f" % (epoch, valid_cost, epoch_toc - epoch_tic))

                logging.info("Validation F1 score: {}, EM score: {}".format(f1, em))

                if len(previous_losses) > 2 and valid_cost > previous_losses[-1]:
                    logging.info("Annealing learning rate by %f" % FLAGS.learning_rate_decay_factor)
                    sess.run(model.learning_rate_decay_op)

                    # however, if em is improved, we still save the model!
                    if em > max(previous_ems):
                        previous_ems.append(em)
                        model.saver.save(sess, checkpoint_path, global_step=epoch)

                    model.saver.restore(sess, checkpoint_path + ("-%d" % best_epoch))
                else:
                    previous_ems.append(em)
                    previous_losses.append(valid_cost)
                    best_epoch = epoch
                    model.saver.save(sess, checkpoint_path, global_step=epoch)
                sys.stdout.flush()
        else:
            # dev mode, we print out validation to "eval" folder
            valid_cost = validate(model, sess, q_valid)

            logging.info("Final Validation cost: %f" % valid_cost)

            # Validate by decoding
            f1, em = decode_validate_logic(model, sess, q_valid, rev_src_vocab, rev_tgt_vocab, rev_env_vocab,
                                           decode_save_dir, FLAGS.best_epoch, sample=5, print_decode=True)
            logging.info("Validation F1 score: {}, EM score: {}".format(f1, em))

def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
