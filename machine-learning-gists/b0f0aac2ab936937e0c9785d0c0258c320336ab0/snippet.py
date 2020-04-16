from mxnet import nd, io

class RBM(object):
    """Bernoulli Restricted Boltzmann Machine (RBM) implementation using MXNET
    """
    def __init__(self, weights=None, hidden_bias=None, visible_bias=None,):
        assert isinstance(weights, nd.NDArray)
        if visible_bias is not None:
            assert visible_bias.shape == (1, weights.shape[0])
        if hidden_bias is not None:
            assert hidden_bias.shape == (1, weights.shape[1])

        self.weights = weights
        self.visible_bias = visible_bias
        self.hidden_bias = hidden_bias

    def fit(self, train_set, batch_size=10, num_epochs=10, gibbs_sampling_steps=1,
            learning_rate=0.01):
        """ Fit the model to the training data.
        :param train_set: training set
        """
        assert isinstance(train_set, nd.NDArray)
        assert len(train_set.shape) == 2
        assert train_set.shape[1] == self.weights.shape[0]

        train_set = train_set.reshape((train_set.shape[0], 1, train_set.shape[1]))

        for _ in range(num_epochs):
            """ For each epoch shuffle the training set.
            Iteratively do batch training.
            """
            for batch in io.NDArrayIter(data=train_set, shuffle=True,
                                        batch_size=batch_size,
                                        last_batch_handle='discard'):
                self._train_batch(batch.data[0], gibbs_sampling_steps, learning_rate)

    def _train_batch(self, batch, gibbs_sampling_steps, learning_rate):
        """Performs k-step Contrastive Divergence (CD-k) learning.
        Updates weights and biases.
        Keep in mind that most variables are "batch" tensors.
        Variable name suffix "_pr" stands for Pr. (probability).
        """
        hidden_pr, hidden, dreamed_visible, dreamed_hidden_pr = self.gibbs_sampling_step(batch)

        positive_phase = nd.batch_dot(self._transpose_batch(batch), hidden)
        for _ in range(gibbs_sampling_steps - 1):
            _, _, dreamed_visible, dreamed_hidden_pr = self.gibbs_sampling_step(dreamed_visible)
        negative_phase = nd.batch_dot(self._transpose_batch(dreamed_visible), dreamed_hidden_pr)

        #  make learning rate independent from the batch size
        learning_rate = learning_rate / batch.shape[0]

        self.weights += learning_rate * nd.sum(positive_phase - negative_phase, axis=(0,))

        if self.hidden_bias is not None:
            self.hidden_bias += learning_rate * nd.sum(hidden_pr - dreamed_hidden_pr, axis=(0,))
        if self.visible_bias is not None:
            self.visible_bias += learning_rate * nd.sum(batch - dreamed_visible, axis=(0,))

    @staticmethod
    def _transpose_batch(vectors_batch):
        """ Transposes a batch of rows to a batch of columns and v/v without copying.
        :param vectors_batch: batch of columns or vectors.
        :return: a reshaped batch that shares memory with the original one.
        """
        shape = vectors_batch.shape
        assert 1 in shape[1:]
        return vectors_batch.reshape((shape[0], shape[2], shape[1]))

    def gibbs_sampling_step(self, visible):
        """ Performs one step of Gibbs sampling.
        :param visible: activations of the visible units
        :return: tuple(hidden state probability, hidden state, visible state probability,
        dreamed hidden state probability, dreamed hidden state)
        """
        hidden_pr, hidden = self._sample_hidden_from_visible(visible)
        dreamed_visible = self._sample_visible_from_hidden(hidden_pr)
        dreamed_hidden_pr, _ = self._sample_hidden_from_visible(dreamed_visible)

        return hidden_pr, hidden, dreamed_visible, dreamed_hidden_pr

    def _sample_hidden_from_visible(self, visible):
        """ Sample the hidden units from the visible units.
        This is the Positive phase of the 1-step Contrastive Divergence algorithm.
        :param visible: activations of the visible units
        :return: tuple(hidden state probability, hidden state)
        """
        batch_size = visible.shape[0]
        activations = nd.batch_dot(visible, self._broadcast_to_batch(self.weights, batch_size))
        if self.hidden_bias is not None:
            activations += self._broadcast_to_batch(self.hidden_bias, batch_size)
        hidden_pr = nd.Activation(activations, act_type="sigmoid")
        hidden = self._sample_bernoulli(hidden_pr)

        return hidden_pr, hidden

    def _sample_visible_from_hidden(self, hidden):
        """ Sample the visible units from the hidden units.
        This is the Negative phase of the 1-step Contrastive Divergence algorithm.
        :param hidden: activations of the hidden units
        :return: dreamed visible state probability
        """
        batch_size = hidden.shape[0]
        activations = self._transpose_batch(nd.batch_dot(
            self._broadcast_to_batch(self.weights, batch_size),
            self._transpose_batch(hidden)
        ))
        if self.visible_bias is not None:
            activations += self._broadcast_to_batch(self.visible_bias, batch_size)
        dreamed_visible = nd.Activation(activations, act_type="sigmoid")

        return dreamed_visible

    @staticmethod
    def _sample_bernoulli(probability):
        return nd.greater(probability, nd.uniform(shape=probability.shape))
        #  return nd.sign(1 + nd.sign(probability - nd.uniform(shape=probability.shape)))

    @staticmethod
    def _broadcast_to_batch(matrix, batch_size):
        return matrix.broadcast_to(shape=(batch_size, ) + matrix.shape)

    def predict(self, batch):
        """todo"""
        assert isinstance(batch, nd.NDArray)
        assert batch.shape[1:] == (1, self.weights.shape[0])

        _, _, dreamed_visible, _ = self.gibbs_sampling_step(batch)
        return dreamed_visible
