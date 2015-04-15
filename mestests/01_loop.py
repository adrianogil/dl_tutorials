import theano
from theano import tensor as T
import numpy

# x will be the sequence we iterate on
x = T.tensor3('x')
# axes are (time, batch, dimension)
# h0 will be the initial state
h0 = T.zeros((x.shape[1], x.shape[2]))
# c will be a context
c = T.matrix('c')

fn = lambda x, h, c: 0*x + h + c
h, updates = theano.scan(fn=fn, sequences=[x], 
                           outputs_info=[h0],
                           non_sequences=[c])

h.eval({x:numpy.ones((10, 10, 1)),
c:numpy.ones((10,1)).reshape((10, 1))})

# Duplicate sequence
# h.eval({x:numpy.ones((10, 10, 1)),
# c:numpy.ones((10,1)).reshape((10, 1))})