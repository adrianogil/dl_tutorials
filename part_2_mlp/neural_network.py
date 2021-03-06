import numpy
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid
from blocks.utils import shared_floatx_zeros
floatX = theano.config.floatX


def randomWeightInitialize(W):
    W.set_value(
        0.1 * (numpy.random.uniform(
                size=W.get_value().shape
                ).astype(floatX) - 0.5
            )
        )
    return W

class NeuralNetwork(object):
    def __init__(self, input_dim, n_hidden):
        self.input_dim = input_dim
        self.n_hidden = n_hidden

        self.neural_arch = [input_dim] + n_hidden

        self.W = []
        self.b = []

        # Creating weights for each layer (except the last one)
        for n in xrange(len(self.neural_arch) - 1):
            n_in = self.neural_arch[n]
            n_out = self.neural_arch[n+1]

            W = shared_floatx_zeros((n_in, n_out))
            randomWeightInitialize(W)
            W.name = 'W_' + str(n)

            b = shared_floatx_zeros(n_out)
            b.name = 'b_' + str(n)

            self.W.append(W)
            self.b.append(b)

        # Creating last layer
        W = shared_floatx_zeros((n_hidden[-1], 1))
        randomWeightInitialize(W)
        W.name = 'W_out' 

        b = shared_floatx_zeros(1)
        b.name = 'b_out'

        self.W.append(W)
        self.b.append(b)

        self.params = self.W + self.b

    def get_probs(self, features):
        """Output the probability of being a positive.

        Parameters
        ----------
        features : :class:`~tensor.TensorVariable`
            The features that you consider as input.
            Must have shape (batch_size, input_dim).

        Returns
        -------
        probs : :class:`~tensor.TensorVariable`
            The probabilities of each example to belong to
            the positive class. Must have shape (batch_size, 1)
        """
        out = features;

        for n in xrange(len(self.neural_arch)):
            out = sigmoid(out.dot(self.W[n]) + self.b[n])

        return out

    def get_params(self):
        """Returns the list of parameters of the class.

        Returns
        -------
        params : list
            The list of shared variables that are parameters of the class.
        """
        return self.params

    def get_cost(self, probs, targets):
        """Output the logistic loss.

        Parameters
        ----------
        probs : :class:`~tensor.TensorVariable`
            The probabilities of each example to belong to
            the positive class. Must have shape (batch_size, 1)
        targets : :class:`~tensor.TensorVariable`
            The indicator on whether the example belongs to the
            positive class. Must have shape (batch_size, 1)

        Returns
        -------
        cost : :class:`~tensor.TensorVariable`
            The corresponding logistic cost.
            .. math:: -targets \log(probs) - (1 - targets) \log(1 - probs)
        """
        return -targets * T.log(probs) - (1 - targets) * T.log(1 - probs)

    def get_misclassification(self, probs, targets):
        """Output the misclassification error.

        This misclassification is done when classifying an example as
        the most likely class.

        Parameters
        ----------
        probs : :class:`~tensor.TensorVariable`
            The probabilities of each example to belong to
            the positive class. Must have shape (batch_size, 1)
        targets : :class:`~tensor.TensorVariable`
            The indicator on whether the example belongs to the
            positive class. Must have shape (batch_size, 1)

        Returns
        -------
        misclassification : :class:`~tensor.TensorVariable`
            The corresponding misclassification error, if we classify
            an example as the most likely class.
        """
        return targets * (probs < 0.5) + (1-targets) * (probs > 0.5)
