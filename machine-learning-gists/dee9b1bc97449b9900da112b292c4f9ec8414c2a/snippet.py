# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
import tensorflow as tf

class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
    '''Convolutional LSTM (Long short-term memory unit) recurrent network cell.

    The class uses optional peep-hole connections, optional cell-clipping,
    optional normalization layer, and an optional recurrent dropout layer.

    Basic implmentation is based on tensorflow, tf.nn.rnn_cell.LSTMCell.

    Default LSTM Network implementation is based on:

        http://www.bioinf.jku.at/publications/older/2604.pdf

    Sepp Hochreiter, Jurgen Schmidhuber.
    "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

    Peephole connection is based on:

        https://research.google.com/pubs/archive/43905.pdf

    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for large scale acoustic modeling". 2014.

    Default Convolutional LSTM implementation is based on:

        https://arxiv.org/abs/1506.04214

    Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, Wang-chun Woo.
    "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting". 2015.

    Recurrent dropout is base on:
    
        https://arxiv.org/pdf/1603.05118

    Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth
    "Recurrent Dropout without Memory Loss". 2016.

    Normalization layer is applied prior to nonlinearities.
    '''
    def __init__(self,
                 shape,
                 kernel,
                 depth,
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=None,
                 forget_bias=1.0,
                 activation=None,
                 normalize=None,
                 dropout=None,
                 reuse=None):
        '''Initialize the parameters for a ConvLSTM Cell.

        Args:
            shape: list of 2 integers, specifying the height and width 
                of the input tensor.
            kernel: list of 2 integers, specifying the height and width 
                of the convolutional window.
            depth: Integer, the dimensionality of the output space.
            use_peepholes: Boolean, set True to enable diagonal/peephole connections.
            cell_clip: Float, if provided the cell state is clipped by this value 
                prior to the cell output activation.
            initializer: The initializer to use for the weights.
            forget_bias: Biases of the forget gate are initialized by default to 1
                in order to reduce the scale of forgetting at the beginning of the training.
            activation: Activation function of the inner states. Default: `tanh`.
            normalize: Normalize function, if provided inner states is normalizeed 
                by this function.
            dropout: Float, if provided dropout is applied to inner states 
                with keep probability in this value.
            reuse: Boolean, whether to reuse variables in an existing scope.
        '''
        super(ConvLSTMCell, self).__init__(_reuse=reuse)

        tf_shape = tf.TensorShape(shape + [depth])
        self._output_size = tf_shape
        self._state_size = tf.nn.rnn_cell.LSTMStateTuple(tf_shape, tf_shape)

        self._kernel = kernel
        self._depth = depth
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._forget_bias = forget_bias
        self._activation = activation or tf.nn.tanh
        self._normalize = normalize
        self._dropout = dropout

        self._w_conv = None
        if self._use_peepholes:
            self._w_f_diag = None
            self._w_i_diag = None
            self._w_o_diag = None

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        '''Run one step of ConvLSTM.

        Args:
            inputs: input Tensor, 4D, (batch, shape[0], shape[1], depth)
            state: tuple of state Tensor, both `4-D`, with tensor shape `c_state` and `m_state`.

        Returns:
            A tuple containing:

            - A '4-D, (batch, height, width, depth)', Tensor representing 
                the output of the ConvLSTM after reading `inputs` when previous 
                state was `state`.
                Here height, width is:
                    shape[0] and shape[1].
            - Tensor(s) representing the new state of ConvLSTM after reading `inputs` when
                the previous state was `state`. Same type and shape(s) as `state`.
        '''
        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(4)[3]
        if input_size.value is None:
            raise ValueError('Could not infer size from inputs.get_shape()[-1]')

        c_prev, m_prev = state
        inputs = tf.concat([inputs, m_prev], axis=-1)

        if not self._w_conv:
            scope = tf.get_variable_scope()
            with tf.variable_scope(scope, initializer=self._initializer):
                kernel_shape = self._kernel + [inputs.shape[-1].value, 4 * self._depth]
                self._w_conv = tf.get_variable('w_conv', shape=kernel_shape, dtype=dtype)

        # i = input_gate, j = new_input, f = forget_gate, o = ouput_gate
        conv = tf.nn.conv2d(inputs, self._w_conv, (1, 1, 1, 1), 'SAME')
        i, j, f, o = tf.split(conv, 4, axis=-1)

        # Diagonal connections
        if self._use_peepholes and not self._w_f_diag:
            scope = tf.get_variable_scope()
            with tf.variable_scope(scope, initializer=self._initializer):
                self._w_f_diag = tf.get_variable('w_f_diag', c_prev.shape[1:], dtype=dtype)
                self._w_i_diag = tf.get_variable('w_i_diag', c_prev.shape[1:], dtype=dtype)
                self._w_o_diag = tf.get_variable('w_o_diag', c_prev.shape[1:], dtype=dtype)

        if self._use_peepholes:
            f = f + self._w_f_diag * c_prev
            i = i + self._w_i_diag * c_prev
        if self._normalize is not None:
            f = self._normalize(f)
            i = self._normalize(i)
            j = self._normalize(j)

        j = self._activation(j)

        if self._dropout is not None:
            j = tf.nn.dropout(j, self._dropout)

        c = tf.nn.sigmoid(f + self._forget_bias) * c_prev + tf.nn.sigmoid(i) * j

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            o = o + self._w_o_diag * c
        if self._normalize is not None:
            o = self._normalize(o)
            c = self._normalize(c)

        m = tf.nn.sigmoid(o) * self._activation(c)

        new_state = tf.nn.rnn_cell.LSTMStateTuple(c, m)
        return m, new_state