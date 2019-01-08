from __future__ import absolute_import
import six
import warnings
from . import backend as K
from .utils.generic_utils import deserialize_keras_object
from .engine import Layer
import tensorflow as tf

def my_leakyrelu(x,a=0.01):
    cond = tf.less(x, tf.constant(0))
    return tf.where(cond, a*x, x)

def lrelu001(x,a=0.01):
   return K.maximum(x,a*x)

def lrelu030(x,a=0.30):
   return K.maximum(x,a*x)


#def cube(x):
#    return (x**3.0)
def sin(x):
    return K.sin(x)

def cube(x):
    return K.pow(x,3)

def swish(x):
    beta = 1
    return (K.sigmoid(x*beta) * x)

def penalized_tanh(x):
    alpha = 0.25
    return K.maximum(tanh(x), alpha*tanh(x))

def lrelu(x):
    return K.maximum(x, 0.01*x)

def maxsig(x):
    return K.maximum(x, K.sigmoid(x))

def cosper(x):
    return (K.cos(x) -x)

def minsin(x):
    return K.minimum(x, K.sin(x))

def tanhrev(x):
#    return  (K.pow(K.cos(x)/K.sin(x),2) -x)
    return (K.pow(tf.atan(x),2)-x)


def maxtanh(x):
    return K.maximum(x, K.tanh(x))




def softmax(x, axis=-1):
    """Softmax activation function.

    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.

    # Returns
        Tensor, output of softmax transformation.

    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


def elu(x, alpha=1.0):
    return K.elu(x, alpha)


def selu(x):
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)

    # Arguments
        x: A tensor or variable to compute the activation function for.

    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * K.elu(x, alpha)


def softplus(x):
    return K.softplus(x)


def softsign(x):
    return K.softsign(x)


def relu(x, alpha=0., max_value=None):
    return K.relu(x, alpha=alpha, max_value=max_value)


def tanh(x):
    return K.tanh(x)


def sigmoid(x):
    return K.sigmoid(x)


def hard_sigmoid(x):
    return K.hard_sigmoid(x)


def linear(x):
    return x


def serialize(activation):
    return activation.__name__


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='activation function')


def get(identifier):
    if identifier is None:
        return linear
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        if isinstance(identifier, Layer):
            warnings.warn((
                'Do not pass a layer instance (such as {identifier}) as the '
                'activation argument of another layer. Instead, advanced '
                'activation layers should be used just like any other '
                'layer in a model.'
            ).format(identifier=identifier.__class__.__name__))
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'activation function identifier:', identifier)
