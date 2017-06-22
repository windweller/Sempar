# quick encoder decoder
# data is randomly shuffled, and split into train, val, test

# just need to load them in, no need to shuffle again
# no need to use pair_iter

from data_util import initialize_vocabulary
import time
import logging
import os
import sys

import data_util

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.gen_math_ops import _batch_mat_mul as batch_matmul
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell

from util import exact_match_score, f1_score


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size):
        self.size = size
        self.keep_prob = tf.placeholder(tf.float32)
        self.encoder_cell = rnn_cell.GRUCell(self.size)
        self.encoder_cell = DropoutWrapper(self.encoder_cell, input_keep_prob=self.keep_prob)

    def encode(self, inputs, masks, reuse=False, scope_name=""):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: (batch_size, time_step, size), batch-major
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        with vs.variable_scope(scope_name + "Encoder", reuse=reuse):
            inp = inputs
            mask = masks

            with vs.variable_scope("EncoderCell") as scope:
                srclen = tf.reduce_sum(mask, reduction_indices=0)
                (fw_out, bw_out), (fw_states, bw_states) = tf.nn.bidirectional_dynamic_rnn(self.encoder_cell,
                                                                                           self.encoder_cell, inp,
                                                                                           srclen,
                                                                                           scope=scope,
                                                                                           time_major=False,
                                                                                           dtype=tf.float32)
                out = fw_out + bw_out
                out = self.dropout(out)

            encoder_outputs = self.dropout(fw_states + bw_states)

        return out, encoder_outputs

    def dropout(self, inp):
        return tf.nn.dropout(inp, self.keep_prob)


class GRUCellAttn(rnn_cell.GRUCell):
    def __init__(self, num_units, encoder_output, scope=None):
        self.hs = encoder_output
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn1"):
                hs2d = tf.reshape(self.hs, [-1, num_units])
                phi_hs2d = tf.nn.tanh(rnn_cell._linear(hs2d, num_units, True, 1.0))
                self.phi_hs = tf.reshape(phi_hs2d, tf.shape(self.hs))
        super(GRUCellAttn, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        gru_out, gru_state = super(GRUCellAttn, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn2"):
                gamma_h = tf.nn.tanh(rnn_cell._linear(gru_out, self._num_units, True, 1.0))
            weights = tf.reduce_sum(self.phi_hs * gamma_h, reduction_indices=2, keep_dims=True)
            weights = tf.nn.softmax(weights, dim=1)
            context = tf.reduce_sum(self.hs * weights, reduction_indices=1)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))
            return (out, out)


class Decoder(object):
    def __init__(self, size, num_layers, output_size=None, attn=False):

        self.size = size
        self.num_layers = num_layers
        self.output_size = size if output_size is None else output_size
        self.attn = attn

        self.keep_prob = tf.placeholder(tf.float32)

        self.decoder_state_input, self.decoder_state_output = [], []
        for i in xrange(num_layers):
            self.decoder_state_input.append(tf.placeholder(tf.float32, shape=[None, size]))

    def dropout(self, inp):
        return tf.nn.dropout(inp, self.keep_prob)

    def decode(self, decoder_inputs, encoder_output, target_length, reuse=False):

        if self.num_layers > 1:
            self.decoder_cell = rnn_cell.GRUCell(self.size)

        if self.attn:
            self.attn_cell = GRUCellAttn(self.size, encoder_output, scope="DecoderAttnCell")
        else:
            self.attn_cell = rnn_cell.GRUCell(self.size)

        i = -1
        with vs.variable_scope("Decoder", reuse=reuse):
            inp = decoder_inputs
            for i in xrange(self.num_layers - 1):
                with vs.variable_scope("DecoderCell%d" % i) as scope:
                    out, state_output = rnn.dynamic_rnn(self.decoder_cell, inp, time_major=False,
                                                        dtype=tf.float32, sequence_length=target_length,
                                                        scope=scope, initial_state=self.decoder_state_input[i])
                    inp = self.dropout(out)
                    self.decoder_state_output.append(state_output)

            with vs.variable_scope("DecoderAttnCell") as scope:
                out, state_output = rnn.dynamic_rnn(self.attn_cell, inp, time_major=False,
                                                    dtype=tf.float32, sequence_length=target_length,
                                                    scope=scope, initial_state=self.decoder_state_input[i + 1])
                decoder_output = self.dropout(out)
                self.decoder_state_output.append(state_output)

        # (batch_size, T, hidden_size)
        return decoder_output

    def set_default_decoder_state_input(self, input_feed, batch_size):
        default_value = np.zeros([batch_size, self.size])
        for i in xrange(self.num_layers):
            input_feed[self.decoder_state_input[i]] = default_value


def rnn_linear(all_states, dim, output_size, scope, reuse=False, return_param=False):
    with tf.variable_scope(scope, reuse=reuse) as v_s:
        # all_states: (batch_size, time, hidden_size)
        doshape = tf.shape(all_states)
        batch_size, unroll = doshape[0], doshape[1]

        flattened = tf.reshape(all_states, [-1, dim])
        result2d = rnn_cell._linear(flattened, output_size=output_size, bias=True)
        result3d = tf.reshape(result2d, tf.pack([batch_size, unroll, -1]))

    if return_param:
        linear_params = [v for v in tf.global_variables() if v.name.startswith(v_s.name)]
        return result3d, linear_params

    return result3d


def add_sos_eos(tokens):
    return map(lambda token_list: [data_util.SOS_ID] + token_list + [data_util.EOS_ID], tokens)


def padded(tokens, batch_pad=0):
    maxlen = max(map(lambda x: len(x), tokens)) if batch_pad == 0 else batch_pad
    return map(lambda token_list: token_list + [data_util.PAD_ID] * (maxlen - len(token_list)), tokens)


def pair_iter(q, batch_size, inp_len, query_len):
    # use inp_len, query_len to filter list
    batched_input = []
    batched_query = []
    iter_q = q[:]

    while len(iter_q) > 0:
        while len(batched_input) < batch_size and len(iter_q) > 0:
            pair = iter_q.pop(0)
            if len(pair[0]) <= inp_len and len(pair[1]) <= query_len:
                batched_input.append(pair[0])
                batched_query.append(pair[1])

        padded_input = np.array(padded(batched_input), dtype=np.int32)
        input_mask = (padded_input != data_util.PAD_ID).astype(np.int32)
        batched_query = add_sos_eos(batched_query)
        padded_query = np.array(padded(batched_query), dtype=np.int32)
        query_mask = (padded_query != data_util.PAD_ID).astype(np.int32)

        yield padded_input, input_mask, padded_query, query_mask
        batched_input, batched_query = [], []


class Seq2SeqNeuralParser(object):
    def __init__(self, encoder, decoder, max_gradient_norm, learning_rate,
                 learning_rate_decay_factor, dropout, input_len, src_vocab, tgt_vocab,
                 FLAGS, optimizer="adam"):
        self.inp_len = input_len
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.query_len = FLAGS.query_len
        self.batch_size = FLAGS.batch_size

        self.encoder = encoder
        self.decoder = decoder

        self.inp_tokens = tf.placeholder(tf.int32, shape=[None, None], name="inp_tokens")
        self.inp_mask = tf.placeholder(tf.int32, shape=[None, None], name="inp_masks")

        self.query_tokens = tf.placeholder(tf.int32, shape=[None, None], name="query_tokens")
        self.query_mask = tf.placeholder(tf.int32, shape=[None, None], name="query_mask")

        self.keep_prob_config = 1.0 - dropout
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        self.FLAGS = FLAGS

        # assemble pieces here
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # set up updates
        params = tf.trainable_variables()
        opt = get_optimizer(optimizer)(self.learning_rate)

        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.gradient_norm = tf.global_norm(gradients)
        self.param_norm = tf.global_norm(params)
        self.updates = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)

    def setup_embeddings(self):
        self.L_enc = tf.get_variable("L_enc", [len(self.src_vocab), self.encoder.size])
        self.L_dec = tf.get_variable("L_dec", [len(self.tgt_vocab), self.decoder.size])

        self.encoder_inputs = embedding_ops.embedding_lookup(self.L_enc, self.inp_tokens)
        self.decoder_inputs = embedding_ops.embedding_lookup(self.L_dec, self.query_tokens)

    def setup_system(self):
        self.encoder_output, input_last_state = self.encoder.encode(self.encoder_inputs, self.inp_mask)
        target_length = tf.reduce_sum(self.query_mask, reduction_indices=1)
        self.decoder_output = self.decoder.decode(self.decoder_inputs, self.encoder_output, target_length)

    def setup_loss(self):
        with vs.variable_scope("Logistic"):
            self.outputs = rnn_linear(self.decoder_output,
                                      self.decoder.size, output_size=len(self.tgt_vocab),
                                      scope="flogits", return_param=False)

            unroll = tf.shape(self.outputs)[1]
            batch_size = tf.shape(self.outputs)[0]

            self.seq_loss = tf.nn.seq2seq.sequence_loss_by_example(
                [tf.reshape(self.outputs, [-1, len(self.tgt_vocab)])],
                [tf.reshape(self.query_tokens, [-1])],
                [tf.ones(tf.pack([batch_size * unroll]))])  # tf.cast(tf.reshape(self.query_mask, [-1]), tf.float32)

            self.loss = tf.reduce_sum(self.seq_loss) / self.batch_size

    def optimize(self, session, inp_tokens, inp_mask, query_tokens, query_mask):
        input_feed = {}
        input_feed[self.inp_tokens] = inp_tokens
        input_feed[self.inp_mask] = inp_mask
        input_feed[self.query_tokens] = query_tokens
        input_feed[self.query_mask] = query_mask

        input_feed[self.encoder.keep_prob] = self.keep_prob_config
        input_feed[self.decoder.keep_prob] = self.keep_prob_config

        output_feed = [self.updates, self.gradient_norm, self.loss, self.param_norm]

        self.decoder.set_default_decoder_state_input(input_feed, query_tokens.shape[0])

        outputs = session.run(output_feed, input_feed)

        return outputs[1], outputs[2], outputs[3]

    def test(self, session, inp_tokens, inp_mask, query_tokens, query_mask):
        input_feed = {}
        input_feed[self.inp_tokens] = inp_tokens
        input_feed[self.inp_mask] = inp_mask
        input_feed[self.query_tokens] = query_tokens
        input_feed[self.query_mask] = query_mask

        input_feed[self.encoder.keep_prob] = 1.
        input_feed[self.decoder.keep_prob] = 1.

        output_feed = [self.loss]

        self.decoder.set_default_decoder_state_input(input_feed, query_tokens.shape[0])

        outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def validate(self, session, q_valid):
        valid_costs = []
        for inp_tokens, inp_mask, query_tokens, query_mask in pair_iter(q_valid, self.batch_size, self.inp_len, self.query_len):
            cost = self.test(session, inp_tokens, inp_mask, query_tokens, query_mask)
            valid_costs.append(cost / inp_tokens.shape[0])
        valid_cost = np.sum(valid_costs) / float(len(valid_costs))
        return valid_cost

    def decode_greedy_batch(self, sess, encoder_output, batch_size):
        decoder_state = None
        decoder_input = np.array([data_util.SOS_ID, ] * batch_size, dtype=np.int32)

        output_sent = np.array([data_util.PAD_ID, ] * self.query_len
                               * batch_size, dtype=np.int32).reshape([batch_size, self.query_len])
        dones = np.array([True, ] * self.batch_size, dtype=np.bool)
        i = 0
        while True:
            decoder_output, decoder_state = self.decode(sess, encoder_output, decoder_input,
                                                                   decoder_states=decoder_state)

            # decoder_output shape: (1, batch_size, vocab_size)
            token_highest_prob = np.argmax(np.squeeze(decoder_output), axis=1)

            # token_highest_prob shape: (batch_size,)
            mask = token_highest_prob == data_util.EOS_ID
            update_dones_indices = np.nonzero(mask)
            # update on newly finished sentence, add EOS_ID
            new_finished = update_dones_indices != dones
            output_sent[i, new_finished] = data_util.EOS_ID

            dones[update_dones_indices] = False
            if i >= self.query_len - 1 or np.sum(np.nonzero(dones)) == 0:
                break

            output_sent[i, dones] = token_highest_prob
            decoder_input = token_highest_prob.reshape([1, batch_size])
            i += 1

        return output_sent

    def get_encode(self, session, inp_tokens, inp_mask):
        input_feed = {}
        input_feed[self.inp_tokens] = inp_tokens
        input_feed[self.inp_mask] = inp_mask
        input_feed[self.encoder.keep_prob] = 1.

        output_feed = [self.encoder_output]

        outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def decode(self, session, inp_tokens, query_tokens, query_mask=None, decoder_states=None):
        input_feed = {}
        input_feed[self.inp_tokens] = inp_tokens
        input_feed[self.query_tokens] = query_tokens
        input_feed[self.query_mask] = query_mask if query_mask is not None else np.ones_like(query_tokens)

        input_feed[self.encoder.keep_prob] = 1.
        input_feed[self.decoder.keep_prob] = 1.

        output_feed = [self.outputs] + + self.decoder.decoder_state_output

        if decoder_states is None:
            self.decoder.set_default_decoder_state_input(input_feed, query_tokens.shape[0])
        else:
            for i in xrange(self.decoder.num_layers):
                input_feed[self.decoder.decoder_state_input[i]] = decoder_states[i]

        outputs = session.run(output_feed, input_feed)

        return outputs[0], outputs[1:]

    def evaluate_answer(self, session, q, rev_src_vocab, rev_tgt_vocab, sample=100, print_every=100):
        # this is teacher-forcing evaluation, not even greedy decode
        f1 = 0.
        em = 0.
        size = 0.

        for inp_tokens, inp_mask, query_tokens, query_mask in pair_iter(q, self.batch_size, self.inp_len,
                                                                        self.query_len):
            # decoder_output = self.decode(session, inp_tokens, inp_mask, query_tokens, query_mask)
            encoder_output = self.get_encode(session, inp_tokens, inp_mask)
            decoder_output = self.decode_greedy_batch(session, encoder_output, self.batch_size)

            # decoder_tokens = np.argmax(decoder_output, axis=-1)
            decoder_tokens = decoder_output

            # those are batched right now
            # decoder_tokens = np.squeeze(decoder_tokens) * query_mask
            # query_tokens = query_tokens * query_mask

            batch_size = inp_tokens.shape[0]
            query_len = np.sum(query_mask, axis=1)

            for i in range(batch_size):
                f1 += f1_score(str(decoder_tokens[i, 1:query_len[i]-1]), str(query_tokens[i, 1:query_len[i]-1]))
                em += exact_match_score(str(decoder_tokens[i, 1:query_len[i]-1]), str(query_tokens[i, 1:query_len[i]-1]))

                size += 1

                if size % print_every == 0:
                    decoded_parse = [rev_tgt_vocab[j] for j in decoder_tokens[i, 1:-1] if j != data_util.PAD_ID]
                    true_parse = [rev_tgt_vocab[j] for j in query_tokens[i, 1:-1] if j != data_util.PAD_ID]
                    decoded_input = [rev_src_vocab[j] for j in inp_tokens[i, :] if j != data_util.PAD_ID]
                    print("input: {}".format(" ".join(decoded_input)))
                    print("decoded result: {}".format(" ".join(decoded_parse)))
                    print("ground truth result: {}".format(" ".join(true_parse)))

            if size >= sample:
                break

        f1 /= size
        em /= size

        return f1, em

    def train(self, session, q_train, q_valid, rev_src_vocab, rev_tgt_vocab, num_epochs, save_train_dir):
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        epoch = 0
        best_epoch = 0
        previous_losses = []
        exp_cost = None
        exp_length = None
        exp_norm = None

        while num_epochs == 0 or epoch < num_epochs:
            epoch += 1
            current_step = 0

            ## Train
            epoch_tic = time.time()
            for inp_tokens, inp_mask, query_tokens, query_mask in pair_iter(q_train, self.batch_size, self.inp_len, self.query_len):
                # Get a batch and make a step.
                tic = time.time()

                grad_norm, cost, param_norm = self.optimize(session, inp_tokens, inp_mask,
                                                            query_tokens, query_mask)

                toc = time.time()
                iter_time = toc - tic
                current_step += 1

                lengths = np.sum(query_mask, axis=1)
                mean_length = np.mean(lengths)

                if not exp_cost:
                    exp_cost = cost
                    exp_length = mean_length
                    exp_norm = grad_norm
                else:
                    exp_cost = 0.99 * exp_cost + 0.01 * cost
                    exp_length = 0.99 * exp_length + 0.01 * mean_length
                    exp_norm = 0.99 * exp_norm + 0.01 * grad_norm

                cost = cost / mean_length

                if current_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, cost %f, exp_cost %f, grad norm %f, param norm %f, batch time %f' %
                        (epoch, current_step, cost, exp_cost / exp_length, grad_norm, param_norm, iter_time))

            epoch_toc = time.time()

            ## Checkpoint
            checkpoint_path = os.path.join(save_train_dir, "sempar.ckpt")

            ## Validate
            valid_cost = self.validate(session, q_valid)

            logging.info("Epoch %d Validation cost: %f epoch time: %f" % (epoch, valid_cost, epoch_toc - epoch_tic))

            ## Measure F1 and EM
            f1, em = self.evaluate_answer(session, q_train, rev_src_vocab, rev_tgt_vocab, sample=5, print_every=2)
            logging.info("Training F1 score: {}, EM score: {}".format(f1, em))

            f1, em = self.evaluate_answer(session, q_valid, rev_src_vocab, rev_tgt_vocab, sample=5, print_every=2)
            logging.info("Validation F1 score: {}, EM score: {}".format(f1, em))

            if epoch >= self.FLAGS.learning_rate_decay_epoch:
                logging.info("Annealing learning rate after epoch {} by {}".format(self.FLAGS.learning_rate_decay_epoch,
                                                                                   self.FLAGS.learning_rate_decay_factor))
                session.run(self.learning_rate_decay_op)

            if len(previous_losses) > 2 and valid_cost > previous_losses[-1]:
                # logging.info("Additional annealing learning rate by %f" % self.FLAGS.learning_rate_decay_factor)
                # session.run(self.learning_rate_decay_op)
                logging.info("validation cost trigger: restore model from epoch %d" % best_epoch)
                self.saver.restore(session, checkpoint_path + ("-%d" % best_epoch))
            else:
                previous_losses.append(valid_cost)
                best_epoch = epoch
                self.saver.save(session, checkpoint_path, global_step=epoch)

