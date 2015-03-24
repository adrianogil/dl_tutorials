import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid
from blocks.utils import shared_floatx_zeros

import numpy as np

class LogisticRegressor(object):

    def __init__(self, input_dim):

        # WRITEME
        self.input_dim = input_dim
        self.params = [shared_floatx_zeros((input_dim, 1)),
                       shared_floatx_zeros((1,))]
        self.w = self.params[0]
        self.b = self.params[1]

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

        return T.nnet.sigmoid(theano.dot(features, self.w) + self.b)


    def get_params(self):
        """Returns the list of parameters of the model.

        Returns
        -------
        params : list
            The list of shared variables that are parameters of the model.
        """
        # WRITEME
        #pass
        return [self.w, self.b]

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
        # WRITEME
        # pass
        return  (-targets) * T.log(probs) - (1 - targets) * T.log(1 - probs)


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

        # WRITEME
        #pass
        
        return targets * (probs < 0.5) + (1 - targets) * (probs > 0.5) 
