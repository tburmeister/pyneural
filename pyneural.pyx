import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

import time

np.import_array()

cdef extern from "neural.h":
    struct neural_net_layer:
        float *bias
        float *theta
        float *act
        float *delta
        neural_net_layer *prev
        neural_net_layer *next
        int in_nodes
        int out_nodes  

    void neural_feed_forward(neural_net_layer *head, float *x)

    void neural_back_prop(neural_net_layer *tail, float *y, 
            const float alpha, const float lamb)

    void neural_sgd_iteration(neural_net_layer *head, neural_net_layer *tail,
            float *features, float *labels, const int n_samples, 
            const float alpha, const float lamb)

cdef class NetLayer:
    cdef neural_net_layer _net_layer
    cdef NetLayer prev, next
    cdef np.ndarray bias, theta, act, delta

    def __init__(self, in_nodes, out_nodes):
        bias = 0.24 * np.random.rand(out_nodes) - 0.12
        self.bias = bias.copy(order='C').astype(np.float32)
        self._net_layer.bias = <float *>np.PyArray_DATA(self.bias)

        theta = 0.24 * np.random.rand(out_nodes * in_nodes) - 0.12
        self.theta = theta.copy(order='C').astype(np.float32)
        self._net_layer.theta = <float *>np.PyArray_DATA(self.theta)

        self.act = np.zeros(in_nodes).astype(np.float32)
        self._net_layer.act = <float *>np.PyArray_DATA(self.act)

        self.delta = np.zeros(in_nodes).astype(np.float32)
        self._net_layer.delta = <float *>np.PyArray_DATA(self.delta)

        self._net_layer.in_nodes = in_nodes
        self._net_layer.out_nodes = out_nodes

    cdef void set_prev(self, NetLayer prev):
        self.prev = prev
        self._net_layer.prev = &prev._net_layer

    cdef void set_next(self, NetLayer next):
        self.next = next
        self._net_layer.next = &next._net_layer

    cdef neural_net_layer *get_ptr(self):
        return &self._net_layer

cdef class NeuralNet:
    cdef int n_features, n_labels, n_nodes, n_layers
    cdef NetLayer head, tail

    def __init__(self, n_features, n_labels, n_nodes, n_layers):
        self.n_features = n_features
        self.n_labels = n_labels
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.head = None
        self.tail = None
        self.random_init()

    def random_init(self):
        self.head = NetLayer(self.n_features, self.n_nodes)
        prev = self.head

        for k in xrange(self.n_layers - 1):
            curr = NetLayer(self.n_nodes, self.n_nodes)
            curr.set_prev(prev)
            prev.set_next(curr)
            prev = curr

        curr = NetLayer(self.n_nodes, self.n_labels)
        curr.set_prev(prev)
        prev.set_next(curr)
        prev = curr

        self.tail = NetLayer(self.n_labels, 0)
        self.tail.set_prev(prev)
        prev.set_next(self.tail)

    def _feed_forward(self, np.ndarray[float, ndim=2, mode="c"] x not None):
        neural_feed_forward(self.head.get_ptr(), <float *>np.PyArray_DATA(x))

    def _back_prop(self, np.ndarray[float, ndim=2, mode="c"] y not None, alpha, lamb):
        neural_back_prop(self.tail.get_ptr(), <float *>np.PyArray_DATA(y), alpha, lamb)

    def _sgd_iteration(self, np.ndarray[float, ndim=2, mode="c"] features not None, 
                       np.ndarray[float, ndim=2, mode="c"] labels not None, 
                       alpha, lamb):
        neural_sgd_iteration(self.head.get_ptr(), self.tail.get_ptr(), 
                <float *>np.PyArray_DATA(features), <float *>np.PyArray_DATA(labels), 
                features.shape[0], alpha, lamb)

    def train(self, features, labels, max_iter, alpha, lamb, decay):
        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert features.shape[0] == labels.shape[0]
        assert features.shape[1] == self.n_features
        assert labels.shape[1] == self.n_labels
        _features = features.copy(order='C').astype(np.float32)
        _labels = labels.copy(order='C').astype(np.float32)

        for i in xrange(max_iter):
            start = time.time()
            # TODO: shuffle
            self._sgd_iteration(_features, _labels, alpha, lamb)
            end = time.time()
            print "iteration %d completed in %f seconds" % (i, end - start)
            alpha *= decay
