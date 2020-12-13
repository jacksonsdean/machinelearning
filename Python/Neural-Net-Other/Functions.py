import numpy as np
from random import random
from math import log

def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_prime(value):
    return value * (1.0 - value)


def heavyside(value):
    return 1 * (value > 0)


def heavyside_prime(value):
    return value


def relu(x):
    return max(0, x)


def relu_prime(x):
    if x <= 0:
        return 0
    else:
        return 1

def leaky_relu(x):
    return max(0.01, x)

def leaky_relu_prime(x):
    if x <= 0:
        return 0.01
    else:
        return 1





def softmax(X):
    return np.exp(X)/np.sum(np.exp(X))


def dropout(x, rate):
    if random() < rate:
        return 0
    else:
        return linear(x)


def exp(x):
    return x * x


def cross_entropy(y_hat, y):
    if y == 1:
      return -log(y_hat)
    else:
      return -log(1 - y_hat)


fn_derivatives = {sigmoid:      sigmoid_prime,
                  heavyside:    linear,
                  relu:         relu_prime,
                  leaky_relu:   leaky_relu_prime,
                  dropout:      linear,
                  linear:       linear}


