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

    cdef void set_prev(self, prev):
        self.prev = prev
        self._net_layer.prev = &(prev._net_layer)

    cdef void set_next(self, next):
        self.next = next
        self._net_layer.next = &(next._net_layer)

    cdef neural_net_layer *get_ptr(self):
        return &(self._net_layer)

cdef neural_net_layer *_random_layer(int in_nodes, int out_nodes):
    cdef neural_net_layer *layer = <neural_net_layer *>malloc(sizeof(neural_net_layer))

    bias = 0.24 * np.random.rand(out_nodes) - 0.12
    bias = bias.copy(order='C').astype(np.float32)
    layer.bias = <float *>np.PyArray_DATA(bias)

    theta = 0.24 * np.random.rand(out_nodes * in_nodes) - 0.12
    theta = theta.copy(order='C').astype(np.float32)
    layer.theta = <float *>np.PyArray_DATA(theta)

    act = np.zeros(in_nodes).astype(np.float32)
    layer.act = <float *>np.PyArray_DATA(act)

    delta = np.zeros(in_nodes).astype(np.float32)
    layer.delta = <float *>np.PyArray_DATA(delta)

    layer.in_nodes = in_nodes
    layer.out_nodes = out_nodes
    return layer

cdef class NeuralNet:
    cdef int n_features, n_labels, n_nodes, n_layers

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
            print "curr layer %d %d" % (curr.in_nodes, curr.out_nodes)
            curr.set_prev(prev)
            prev.set_next(curr)
            prev = curr

        curr = NetLayer(self.n_nodes, self.n_labels)
        print "curr layer %d %d" % (curr.in_nodes, curr.out_nodes)
        curr.set_prev(prev)
        print "prev layer %d %d" % (curr.prev.in_nodes, curr.prev.out_nodes)
        print "prev layer %d %d" % (prev.in_nodes, prev.out_nodes)
        print "head layer %d %d" % (self.head.in_nodes, self.head.out_nodes)
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

        print "features %d" % self.n_features
        print "labels %d" % self.n_labels
        print "nodes %d" % self.n_nodes
        print "layers %d" % self.n_layers

        for i in xrange(max_iter):
            start = time.time()
            # TODO: shuffle
            self._sgd_iteration(_features, _labels, alpha, lamb)
            end = time.time()
            print "iteration %d completed in %d seconds" % (i, end - start)
            alpha *= decay
