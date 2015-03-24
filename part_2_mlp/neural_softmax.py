import numpy
import theano
from theano import tensor as T
from dl_tutorials.utils.softmax import softmax
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

class NeuralSoftmax(object):
    def __init__(self, input_dim, n_hidden, n_classes):
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.neural_arch = [input_dim] + n_hidden + [n_classes]

        self.W = []
        self.b = []

        # Creating weights for each layer
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

        self.params = self.W + self.b

    def get_probs(self, features):
        """Output the probability of belonging to a class

        Parameters
        ----------
        features : :class:`~tensor.TensorVariable`
            The features that you consider as input.
            Must have shape (batch_size, input_dim).

        Returns
        -------
        probs : :class:`~tensor.TensorVariable`
            The probabilities of each example of belonging to
            each class. Must have shape (batch_size, n_classes)
        """
        out = features;

        for n in xrange(len(self.neural_arch)-1):
            out = softmax(out.dot(self.W[n]) + self.b[n])

        return out

    def get_params(self):
        """Returns the list of parameters of the model.

        Returns
        -------
        params : list
            The list of shared variables that are parameters of the model.
        """
        return self.params

    def get_weights(self):
        """Returns the weights parameter of the model.

        Returns
        -------
        weights : :class:`~tensor.sharedvar.SharedVariable`
            The weights of the model connected to the input.
        """
        return self.W[0]

    def get_cost(self, probs, targets):
        """Output the softmax loss.

        Parameters
        ----------
        probs : :class:`~tensor.TensorVariable`
            The probabilities of each example to belong to
            each class. Must have shape (batch_size, n_classes)
        targets : :class:`~tensor.TensorVariable`
            The indicator of the example class.
            Must have shape (batch_size, 1)

        Returns
        -------
        cost : :class:`~tensor.TensorVariable`
            The corresponding logistic cost.
            .. math:: - \log(probs_{targets})
        """

        return -T.log(
                probs[T.arange(probs.shape[0]),
                  targets.flatten()]
                  )

    def get_misclassification(self, probs, targets):
        """Output the misclassification error.

        This misclassification is done when classifying an example as
        the most likely class.

        Parameters
        ----------
        probs : :class:`~tensor.TensorVariable`
            The probabilities of each example to belong to
            each class. Must have shape (batch_size, n_classes)
        targets : :class:`~tensor.TensorVariable`
            The indicator of the example class.
            Must have shape (batch_size, 1)

        Returns
        -------
        misclassification : :class:`~tensor.TensorVariable`
            The corresponding misclassification error, if we classify
            an example as the most likely class.
        """
        return T.neq(probs.argmax(axis=1), targets.flatten())
