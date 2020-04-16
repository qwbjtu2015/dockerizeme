# Daniel J. Rodriguez
# https://github.com/danieljoserodriguez

import numpy as np


# A straight line function where activation is proportional to input
# ( which is the weighted sum from neuron ).

# In mathematics, an identity function, also called an identity relation or
# identity map or identity transformation, is a function that always returns
# the same value that was used as its argument.
#
# https://en.wikipedia.org/wiki/Identity_function
def identity(x):
    return x


# derivative identity
def d_identity(x):
    return 1.0


# bent identity
#
# The Heaviside step function, or the unit step function, usually denoted by
# H or θ (but sometimes u, 1 or 𝟙), is a discontinuous function, named after
# Oliver Heaviside (1850–1925), whose value is zero for negative arguments
# and one for positive arguments
def bent_identity(x):
    return ((np.sqrt(((x ** 2.0) + 1.0)) - 1.0) / 2.0) + x


# derivative bent identity
def d_bent_identity(x):
    return (x / (2.0 * np.sqrt((x ** 2.0) + 1.0))) + 1.0


# also called heaviside step
#
# https://en.wikipedia.org/wiki/Heaviside_step_function
def binary_step(x):
    return 1.0 if x >= 0.0 else 0.0


def perceptron(weights, bias, inputs):
    return 1.0 if (np.dot(weights, inputs) + bias) > 0.0 else 0.0


def smooth_perceptron(weights, bias, inputs):
    return np.dot(weights, inputs) + bias


# Sigmoid takes a real value as input and outputs another value between 0 and 1.
# It’s easy to work with and has all the nice properties of activation functions:
# it’s non-linear, continuously differentiable, monotonic, and has a fixed output range.
#
# https://en.wikipedia.org/wiki/Logistic_function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# derivative sigmoid
def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)


# Tanh squashes a real-valued number to the range [-1, 1]. It’s non-linear. But unlike
# Sigmoid, its output is zero-centered. Therefore, in practice the tanh non-linearity
# is always preferred to the sigmoid non-linearity.
#
# https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent
def tanh(x):
    return np.tanh(x)


# derivative hyperbolic tangent
def d_tanh(x):
    return 1.0 - tanh(x) ** 2.0


# arc tangent
# https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
def arc_tan(x):
    return np.arctan(x)


# derivative arc tangent
def d_arc_tan(x):
    return 1.0 / ((x ** 2.0) + 1.0)


# Inverse hyperbolic sine
# https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#Inverse_hyperbolic_sine
def ar_sinh(x):
    return np.log(x + np.sqrt((x ** 2.0) + 1.0))


# derivative of Inverse hyperbolic sine
def d_ar_sinh(x):
    return 1.0 / (np.sqrt((x ** 2.0) + 1.0))


# elliot sig also called soft sign
# https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.46.7204
def elliot_sig(x):
    return x / (1.0 + abs(x))


# derivative elliot sig / soft sign
def d_elliot_sig(x):
    return 1.0 / ((1.0 + abs(x)) ** 2.0)


# inverse square root unit
# https://arxiv.org/abs/1710.09967
def isru(x, alpha):
    return x / np.sqrt((x + (alpha * (x ** 2.0))))


# derivative inverse square root unit
def d_isru(x, alpha):
    return x / (np.sqrt(1.0 + (alpha * (x ** 2.0))) ** 3.0)


# inverse square root linear unit
# https://arxiv.org/abs/1710.09967
def isrlu(x, alpha):
    return x / (np.sqrt(1.0 + (alpha * (x ** 2.0)))) if x < 0.0 else x


# derivative inverse square root linear unit
def d_isrlu(x, alpha):
    return ((1.0 / np.sqrt(1.0 + alpha * (x ** 2.0))) ** 3.0) if x < 0.0 else 1.0


# square non-linearity
# https://ieeexplore.ieee.org/document/8489043
def sqnl(x):
    if x > 2.0:
        return 1.0
    elif 0.0 <= x <= 2.0:
        return x - ((x ** 2.0) / 4.0)
    elif -2.0 <= x < 0.0:
        return x + ((x ** 2.0) / 4.0)
    elif x < -2.0:
        return -1.0


# derivative square non-linearity
def d_sqnl(x):
    return (1.0 - (x / 2.0)), (1.0 + (x / 2.0))


# A recent invention which stands for Rectified Linear Units. The formula is deceptively
# simple: max(0,z). Despite its name and appearance, it’s not linear and provides the
# same benefits as Sigmoid but with better performance.
#
# https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
def relu(x):
    return x if x >= 0.0 else 0.0


# derivative rectified linear units
def d_relu(x):
    return 1.0 if x >= 0.0 else 0.0


# bipolar relu
# https://arxiv.org/abs/1709.04054
def brelu(x, index):
    return relu(x) if index % 2.0 == 0.0 else -relu(-x)


# derivative bipolar relu
def d_brelu(x, index):
    return d_relu(x) if index % 2.0 == 0.0 else -d_relu(-x)


# LeakyRelu is a variant of ReLU. Instead of being 0 when z<0, a leaky ReLU allows a
# small, non-zero, constant gradient α (Normally, α=0.01). However, the consistency
# of the benefit across tasks is presently unclear.
#
# https://pdfs.semanticscholar.org/367f/2c63a6f6a10b3b64b8729d601e69337ee3cc.pdf
def leaky_relu(x):
    return x if x >= 0.0 else 0.01 * x


# derivative leaky relu
def d_leaky_relu(x):
    return 1.0 if x >= 0.0 else 0.01


# parametric relu makes the coefficient of leakage into a parameter that is
# learned along with other neural network parameters - alpha
#
# https://arxiv.org/abs/1502.01852
def prelu(x, alpha):
    return x if x >= 0.0 else alpha * x


# derivative parametric relu
def d_prelu(x, alpha):
    return 1.0 if x >= 0.0 else alpha


# randomized leaky relu
# https://arxiv.org/abs/1505.00853
def rrelu(x, alpha):
    return x if x >= 0.0 else alpha * x


# derivative randomized leaky relu
def d_rrelu(x, alpha):
    return 1.0 if x >= 0.0 else alpha


# Exponential Linear Unit or its widely known name ELU is a function that tend to
# converge cost to zero faster and produce more accurate results. Different to other
# activation functions, ELU has a extra alpha constant which should be positive number
#
# ELU is very similar to RELU except negative inputs. They are both in identity
# function form for non-negative inputs. On the other hand, ELU becomes smooth
# slowly until its output equal to -α whereas RELU sharply smooths.
#
# https://arxiv.org/abs/1511.07289
def elu(x, alpha):
    return x if x > 0.0 else alpha * ((np.e ** 2.0) - 1.0)


# derivative exponential linear unit
def d_elu(x, alpha):
    return 1.0 if x > 0.0 else elu(x, alpha) + alpha


# scaled exponential linear unit
# https://en.wikipedia.org/wiki/Activation_function#cite_note-20
def selu(x, alpha):
    # d = 1.0507
    # if x >= 0:
    #     return x * d
    # else:
    #     return d * alpha * ((np.e ** 2) - 1)
    pass


# derivative scaled exponential linear unit
def d_selu(x, alpha):
    pass


# s-shaped relu
# https://arxiv.org/abs/1512.07030
def srelu():
    pass


# derivative s-shaped relu
def d_srelu():
    pass


# adaptive piecewise linear
# https://arxiv.org/abs/1412.6830
def apl(x, alpha):
    # return np.maximum(0, x) + np.sum(alpha * np.max(0, -x + ))
    pass


# derivative apl
def d_apl():
    pass


# gaussian error linear units
# https://arxiv.org/abs/1606.08415
def gelu(x):
    # return (x * (1 + (x / np.sqrt(2)))) / 2
    pass


# derivative gaussian error linear units
def d_gelu():
    pass


# Softmax function calculates the probabilities distribution of the event over ‘n’ different
# events. In general way of saying, this function will calculate the probabilities of each
# target class over all possible target classes. Later the calculated probabilities will be
# helpful for determining the target class for the given inputs.
#
# https://en.wikipedia.org/wiki/Softmax_function
def soft_max(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# derivative soft max
def d_soft_max(output, trues):
    temp = output - trues
    return temp / len(trues)


# soft plus
# http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf
def soft_plus(x):
    return np.log(1.0 + (np.e ** x))


# derivative soft plus
def d_soft_plus(x):
    return 1.0 / (1.0 + (np.e ** -x))


# soft exponential
# https://arxiv.org/abs/1602.01321
def soft_exponential(x, alpha):
    if alpha < 0.0:
        return -((np.log(1.0 - alpha * (x + alpha))) / alpha)
    elif alpha > 0.0:
        return ((np.e ** (alpha * x)) / alpha) + alpha
    elif alpha == 0.0:
        return x


# derivative soft exponential
def d_soft_exponential(x, alpha):
    return 1.0 / (1.0 - alpha * (alpha + x)) if alpha < 0.0 else np.e ** (alpha * x)


# soft clipping
# https://arxiv.org/abs/1810.11509
def soft_clipping(x, alpha):
    return (1.0 / alpha) * (np.log10((1.0 + (np.e ** (alpha * x))) / (1.0 + (np.e ** (alpha * (x - 1.0))))))


# derivative soft clipping
def d_soft_clipping(x, p):
    a = np.cosh((p * x) / 2.0) ** (-1.0)
    b = np.cosh((p / 2.0) * (1.0 - x)) ** (-1.0)
    return 0.5 * np.sinh(p / 2.0) * a * b


def gaussian_radial_basis(x, c, s):
    return np.exp(-1.0 / (2.0 ** (s ** 2.0)) * (x - c) ** 2.0)


# improvement over relu for deeper networks
def swish(x):
    # return x * (1 + np.exp(-x)) ** -1
    return x / ((np.e ** -x) + 1.0)


# derivative swish
def d_swish(x):
    return ((np.e ** x) * ((np.e ** x) + x + 1.0)) / (((np.e ** x) + 1.0) ** 2.0)


# sinusoid 
# https://arxiv.org/abs/1405.2262
def sinusoid(x):
    return np.sin(x)


# derivative sinusoid
def d_sinusoid(x):
    return np.cos(x)


# sinc
# https://en.wikipedia.org/wiki/Sinc_function
def sinc(x):
    return 1.0 if x == 0.0 else np.sin(x) / x


# derivative sinc
def d_sinc(x):
    return 0.0 if x == 0.0 else (np.cos(x) / x) - (np.sin(x) / (x ** 2.0))


# gaussian
# https://en.wikipedia.org/wiki/Gaussian_function
def gaussian(x):
    return np.e ** (-x ** 2.0)


# derivative gaussian
def d_gaussian(x):
    return -2.0 * x * np.e ** (-x ** 2.0)


# gradient relu
def gradient_relu(a, x):
    _relu = relu(x)
    return np.multiply(a, np.int64(_relu > 0.0))