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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.ops.gen_math_ops import _batch_mat_mul as batch_matmul

_PAD = b"<pad>"
_SOS = b"<sos>"
_EOS = b"<eos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class GRUCellAttn(rnn_cell.GRUCell):
    def __init__(self, num_units, encoder_output, scope=None):
        self.hs = encoder_output
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn1"):
                hs2d = tf.reshape(self.hs, [-1, num_units])
                phi_hs2d = tanh(rnn_cell._linear(hs2d, num_units, True, 1.0))
                self.phi_hs = tf.reshape(phi_hs2d, tf.shape(self.hs))
        super(GRUCellAttn, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        gru_out, gru_state = super(GRUCellAttn, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn2"):
                gamma_h = tanh(rnn_cell._linear(gru_out, self._num_units, True, 1.0))
            weights = tf.reduce_sum(self.phi_hs * gamma_h, reduction_indices=2, keep_dims=True)
            weights = tf.exp(weights - tf.reduce_max(weights, reduction_indices=0, keep_dims=True))
            weights = weights / (1e-6 + tf.reduce_sum(weights, reduction_indices=0, keep_dims=True))
            context = tf.reduce_sum(self.hs * weights, reduction_indices=0)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))
            self.attn_map = tf.squeeze(tf.slice(weights, [0, 0, 0], [-1, -1, 1]))
            return (out, out)


class NLCModel(object):
    def __init__(self, src_vocab_size, tgt_vocab_size, env_vocab_size, size, num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, dropout, FLAGS, forward_only=False, optimizer="adam"):
        self.size = size
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.env_vocab_size = env_vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.keep_prob_config = 1.0 - dropout
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        self.task = FLAGS.task

        self.keep_prob = tf.placeholder(tf.float32)
        self.source_tokens = tf.placeholder(tf.int32, shape=[None, None], name="source_tokens")
        # self.target_tokens = tf.placeholder(tf.int32, shape=[None, None], name="target_tokens")
        self.source_mask = tf.placeholder(tf.int32, shape=[None, None], name="source_mask")
        # self.target_mask = tf.placeholder(tf.int32, shape=[None, None], name="target_mask")

        self.ctx_tokens = tf.placeholder(tf.int32, shape=[None, None], name="ctx_tokens")
        self.pred_tokens = tf.placeholder(tf.int32, shape=[None, None], name="pred_tokens")
        self.ctx_mask = tf.placeholder(tf.int32, shape=[None, None], name="ctx_mask")
        self.pred_mask = tf.placeholder(tf.int32, shape=[None, None], name="pred_mask")

        self.beam_size = tf.placeholder(tf.int32)
        self.target_length = tf.reduce_sum(self.pred_mask, reduction_indices=0)

        self.FLAGS = FLAGS

        self.decoder_state_input, self.decoder_state_output = [], []
        for i in xrange(num_layers):
            self.decoder_state_input.append(tf.placeholder(tf.float32, shape=[None, size]))

        if self.task == "context":
            with tf.variable_scope("CtxPred", initializer=tf.uniform_unit_scaling_initializer(1.0)):
                self.setup_embeddings()
                self.setup_encoder()
                # this should be fine...
                self.encoder_output = self.coattn_encode()
                self.setup_decoder(self.encoder_output)
                self.setup_loss()

                self.setup_beam()
        else:
            raise Exception("unimplemented")

        params = tf.trainable_variables()
        if not forward_only:
            opt = get_optimizer(optimizer)(self.learning_rate)

            gradients = tf.gradients(self.losses, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
            #   self.gradient_norm = tf.global_norm(clipped_gradients)
            self.gradient_norm = tf.global_norm(gradients)
            self.param_norm = tf.global_norm(params)
            self.updates = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)  # write_version=tf.train.SaverDef.V1

    def setup_embeddings(self):
        with vs.variable_scope("embeddings"):
            self.L_enc = tf.get_variable("L_enc", [self.src_vocab_size, self.size])
            # self.L_dec = tf.get_variable("L_dec", [self.tgt_vocab_size, self.size])
            self.L_env = tf.get_variable("L_env", [self.env_vocab_size, self.size])

            self.encoder_inputs = embedding_ops.embedding_lookup(self.L_enc, self.source_tokens)
            # self.target_inputs = embedding_ops.embedding_lookup(self.L_dec, self.target_tokens)
            self.ctx_inputs = embedding_ops.embedding_lookup(self.L_env, self.ctx_tokens)
            self.pred_inputs = embedding_ops.embedding_lookup(self.L_env, self.pred_tokens)

    def setup_encoder(self):
        self.encoder_cell = rnn_cell.GRUCell(self.size)

    def normal_encode(self, input, mask, reuse=False, scope_name=""):
        # note that input: [length, batch_size, dim]
        with vs.variable_scope(scope_name + "Encoder", reuse=reuse):
            inp = input
            out = None
            for i in xrange(self.num_layers):
                with vs.variable_scope("EncoderCell%d" % i) as scope:
                    srclen = tf.reduce_sum(mask, reduction_indices=0)
                    out, _ = self.bidirectional_rnn(self.encoder_cell, inp, srclen, scope=scope)
                    inp = self.dropout(out)
        return out

    def coattn_encode(self):
        # only for task: direct prediction

        # (length, batch_size, dim)
        query_w_matrix = self.normal_encode(self.encoder_inputs, self.source_mask)
        context_w_matrix = self.normal_encode(self.ctx_inputs, self.ctx_mask, reuse=True)

        # can add a query variation here (optional)
        # can take out coattention mix...but by experiment it should be better than no coattention

        # in PA4 it was also time-major

        # batch, p, size
        p_encoding = tf.transpose(context_w_matrix, perm=[1, 0, 2])
        # batch, q, size
        q_encoding = tf.transpose(query_w_matrix, perm=[1, 0, 2])
        # batch, size, q
        q_encoding_t = tf.transpose(query_w_matrix, perm=[1, 2, 0])

        # 2). Q->P Attention
        # [256,25,125] vs [128,125,11]
        A = batch_matmul(p_encoding, q_encoding_t)  # (batch, p, q)
        A_p = tf.nn.softmax(A)

        # 3). P->Q Attention
        # transposed: (batch_size, question, context)
        A_t = tf.transpose(A, perm=[0, 2, 1]) # (batch, q, p)
        A_q = tf.nn.softmax(A_t)

        # 4). Query's context vectors
        C_q = batch_matmul(A_q, p_encoding) # (batch, q, p) * (batch, p, size)
        # (batch, q, size)

        # 5). Paragrahp's context vectors
        q_emb = tf.concat(2, [q_encoding, C_q])
        C_p = batch_matmul(A_p, q_emb)  # (batch, p, q) * (batch, q, size * 2)

        # 6). Linear mix of paragraph's context vectors and paragraph states
        co_att = tf.concat(2, [p_encoding, C_p])  # (batch, p, size * 3)

        # This must be another RNN layer
        # however, if it's just normal attention, we don't need to use a different one
        co_att = tf.transpose(co_att, perm=[1, 0, 2]) # (p, batch, size * 3)
        out = self.normal_encode(co_att, self.ctx_mask, scope_name="Final")

        return out

    def rev_coattn_encode(self):
        pass

    def setup_decoder(self, encoder_output):
        if self.num_layers > 1:
            self.decoder_cell = rnn_cell.GRUCell(self.size)
        self.attn_cell = GRUCellAttn(self.size, encoder_output, scope="DecoderAttnCell")
        i = -1
        with vs.variable_scope("Decoder"):
            inp = self.pred_inputs
            for i in xrange(self.num_layers - 1):
                with vs.variable_scope("DecoderCell%d" % i) as scope:
                    out, state_output = rnn.dynamic_rnn(self.decoder_cell, inp, time_major=True,
                                                        dtype=dtypes.float32, sequence_length=self.target_length,
                                                        scope=scope, initial_state=self.decoder_state_input[i])
                    inp = self.dropout(out)
                    self.decoder_state_output.append(state_output)

            with vs.variable_scope("DecoderAttnCell") as scope:
                out, state_output = rnn.dynamic_rnn(self.attn_cell, inp, time_major=True,
                                                    dtype=dtypes.float32, sequence_length=self.target_length,
                                                    scope=scope, initial_state=self.decoder_state_input[i + 1])
                self.decoder_output = self.dropout(out)
                self.decoder_state_output.append(state_output)

    def decoder_graph(self, decoder_inputs, decoder_state_input):
        decoder_output, decoder_state_output = None, []
        inp = decoder_inputs

        with vs.variable_scope("Decoder", reuse=True):
            i = -1
            for i in xrange(self.num_layers - 1):
                with vs.variable_scope("DecoderCell%d" % i) as scope:
                    inp, state_output = self.decoder_cell(inp, decoder_state_input[i])
                    decoder_state_output.append(state_output)

            with vs.variable_scope("DecoderAttnCell") as scope:
                decoder_output, state_output = self.attn_cell(inp, decoder_state_input[i + 1])
                decoder_state_output.append(state_output)

        return decoder_output, decoder_state_output

    def setup_beam(self):
        time_0 = tf.constant(0)
        beam_seqs_0 = tf.constant([[SOS_ID]])
        beam_probs_0 = tf.constant([0.])

        cand_seqs_0 = tf.constant([[EOS_ID]])
        cand_probs_0 = tf.constant([-3e38])

        state_0 = tf.zeros([1, self.size])
        states_0 = [state_0] * self.num_layers

        def beam_cond(time, beam_probs, beam_seqs, cand_probs, cand_seqs, *states):
            return tf.reduce_max(beam_probs) >= tf.reduce_min(cand_probs)

        def beam_step(time, beam_probs, beam_seqs, cand_probs, cand_seqs, *states):
            batch_size = tf.shape(beam_probs)[0]
            inputs = tf.reshape(tf.slice(beam_seqs, [0, time], [batch_size, 1]), [batch_size])
            decoder_input = embedding_ops.embedding_lookup(self.L_env, inputs)
            decoder_output, state_output = self.decoder_graph(decoder_input, states)

            with vs.variable_scope("Logistic", reuse=True):
                do2d = tf.reshape(decoder_output, [-1, self.size])
                logits2d = rnn_cell._linear(do2d, self.tgt_vocab_size, True, 1.0)
                logprobs2d = tf.nn.log_softmax(logits2d)

            total_probs = logprobs2d + tf.reshape(beam_probs, [-1, 1])
            total_probs_noEOS = tf.concat(1, [tf.slice(total_probs, [0, 0], [batch_size, EOS_ID]),
                                              tf.tile([[-3e38]], [batch_size, 1]),
                                              tf.slice(total_probs, [0, EOS_ID + 1],
                                                       [batch_size, self.tgt_vocab_size - EOS_ID - 1])])

            flat_total_probs = tf.reshape(total_probs_noEOS, [-1])
            beam_k = tf.minimum(tf.size(flat_total_probs), self.beam_size)
            next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)

            next_bases = tf.floordiv(top_indices, self.tgt_vocab_size)
            next_mods = tf.mod(top_indices, self.tgt_vocab_size)

            next_states = [tf.gather(state, next_bases) for state in state_output]
            next_beam_seqs = tf.concat(1, [tf.gather(beam_seqs, next_bases),
                                           tf.reshape(next_mods, [-1, 1])])

            cand_seqs_pad = tf.pad(cand_seqs, [[0, 0], [0, 1]])
            beam_seqs_EOS = tf.pad(beam_seqs, [[0, 0], [0, 1]])
            new_cand_seqs = tf.concat(0, [cand_seqs_pad, beam_seqs_EOS])
            EOS_probs = tf.slice(total_probs, [0, EOS_ID], [batch_size, 1])
            new_cand_probs = tf.concat(0, [cand_probs, tf.reshape(EOS_probs, [-1])])

            cand_k = tf.minimum(tf.size(new_cand_probs), self.beam_size)
            next_cand_probs, next_cand_indices = tf.nn.top_k(new_cand_probs, k=cand_k)
            next_cand_seqs = tf.gather(new_cand_seqs, next_cand_indices)

            return [time + 1, next_beam_probs, next_beam_seqs, next_cand_probs, next_cand_seqs] + next_states

        var_shape = []
        var_shape.append((time_0, time_0.get_shape()))
        var_shape.append((beam_probs_0, tf.TensorShape([None, ])))
        var_shape.append((beam_seqs_0, tf.TensorShape([None, None])))
        var_shape.append((cand_probs_0, tf.TensorShape([None, ])))
        var_shape.append((cand_seqs_0, tf.TensorShape([None, None])))
        var_shape.extend([(state_0, tf.TensorShape([None, self.size])) for state_0 in states_0])
        loop_vars, loop_var_shapes = zip(*var_shape)
        ret_vars = tf.while_loop(cond=beam_cond, body=beam_step, loop_vars=loop_vars, shape_invariants=loop_var_shapes,
                                 back_prop=False)
        #    time, beam_probs, beam_seqs, cand_probs, cand_seqs, _ = ret_vars
        cand_seqs = ret_vars[4]
        cand_probs = ret_vars[3]
        self.beam_output = cand_seqs
        self.beam_scores = cand_probs

    def setup_loss(self):
        with vs.variable_scope("Logistic"):
            doshape = tf.shape(self.decoder_output)
            T, batch_size = doshape[0], doshape[1]
            do2d = tf.reshape(self.decoder_output, [-1, self.size])
            logits2d = rnn_cell._linear(do2d, self.tgt_vocab_size, True, 1.0)
            outputs2d = tf.nn.log_softmax(logits2d)
            self.outputs = tf.reshape(outputs2d, tf.pack([T, batch_size, self.tgt_vocab_size]))

            targets_no_GO = tf.slice(self.pred_tokens, [1, 0], [-1, -1])
            masks_no_GO = tf.slice(self.pred_mask, [1, 0], [-1, -1])
            # easier to pad target/mask than to split decoder input since tensorflow does not support negative indexing
            labels1d = tf.reshape(tf.pad(targets_no_GO, [[0, 1], [0, 0]]), [-1])
            mask1d = tf.reshape(tf.pad(masks_no_GO, [[0, 1], [0, 0]]), [-1])
            losses1d = tf.nn.sparse_softmax_cross_entropy_with_logits(logits2d, labels1d) * tf.to_float(mask1d)
            losses2d = tf.reshape(losses1d, tf.pack([T, batch_size]))
            self.losses = tf.reduce_sum(losses2d) / tf.to_float(batch_size)

    def dropout(self, inp):
        return tf.nn.dropout(inp, self.keep_prob)

    def bidirectional_rnn(self, cell, inputs, lengths, scope=None):
        name = scope.name or "BiRNN"
        # Forward direction
        with vs.variable_scope(name + "_FW") as fw_scope:
            output_fw, output_state_fw = rnn.dynamic_rnn(cell, inputs, time_major=True, dtype=dtypes.float32,
                                                         sequence_length=lengths, scope=fw_scope)
        # Backward direction
        inputs_bw = tf.reverse_sequence(inputs, tf.to_int64(lengths), seq_dim=0, batch_dim=1)
        with vs.variable_scope(name + "_BW") as bw_scope:
            output_bw, output_state_bw = rnn.dynamic_rnn(cell, inputs_bw, time_major=True, dtype=dtypes.float32,
                                                         sequence_length=lengths, scope=bw_scope)

        output_bw = tf.reverse_sequence(output_bw, tf.to_int64(lengths), seq_dim=0, batch_dim=1)

        outputs = output_fw + output_bw
        output_state = output_state_fw + output_state_bw

        return (outputs, output_state)

    def set_default_decoder_state_input(self, input_feed, batch_size):
        default_value = np.zeros([batch_size, self.size])
        for i in xrange(self.num_layers):
            input_feed[self.decoder_state_input[i]] = default_value

    def train_engine(self, session, source_tokens, source_mask, ctx_tokens, ctx_mask, pred_tokens, pred_mask):
        input_feed = {}
        input_feed[self.source_tokens] = source_tokens
        input_feed[self.ctx_tokens] = ctx_tokens
        input_feed[self.pred_tokens] = pred_tokens
        input_feed[self.source_mask] = source_mask
        input_feed[self.ctx_mask] = ctx_mask
        input_feed[self.pred_mask] = pred_mask

        input_feed[self.keep_prob] = self.keep_prob_config
        self.set_default_decoder_state_input(input_feed, pred_tokens.shape[1])

        output_feed = [self.updates, self.gradient_norm, self.losses, self.param_norm]

        outputs = session.run(output_feed, input_feed)

        return outputs[1], outputs[2], outputs[3]

    def test_engine(self, session, source_tokens, source_mask, ctx_tokens, ctx_mask, pred_tokens, pred_mask):
        input_feed = {}
        input_feed[self.source_tokens] = source_tokens
        input_feed[self.ctx_tokens] = ctx_tokens
        input_feed[self.pred_tokens] = pred_tokens

        input_feed[self.source_mask] = source_mask
        input_feed[self.ctx_mask] = ctx_mask
        input_feed[self.pred_mask] = pred_mask

        input_feed[self.keep_prob] = 1.
        self.set_default_decoder_state_input(input_feed, pred_tokens.shape[1])

        output_feed = [self.losses]

        outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def encode_engine(self, session, source_tokens, source_mask, ctx_tokens, ctx_mask):
        input_feed = {}
        input_feed[self.source_tokens] = source_tokens
        input_feed[self.ctx_tokens] = ctx_tokens
        input_feed[self.source_mask] = source_mask
        input_feed[self.ctx_mask] = ctx_mask
        input_feed[self.keep_prob] = 1.

        output_feed = [self.encoder_output]

        outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def decode_beam(self, session, encoder_output, beam_size=8):
        input_feed = {}
        input_feed[self.encoder_output] = encoder_output
        input_feed[self.keep_prob] = 1.
        input_feed[self.beam_size] = beam_size

        output_feed = [self.beam_output, self.beam_scores]

        outputs = session.run(output_feed, input_feed)

        return outputs[0], outputs[1]
