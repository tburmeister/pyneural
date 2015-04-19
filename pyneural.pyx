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

    void neural_sgd_iteration(neural_net_layer *head, neural_net_layer *tail,
            float *features, float *labels, const int n_samples, 
            const float alpha, const float lamb)

    void neural_predict_prob(neural_net_layer *head, neural_net_layer *tail,
            float *features, float *preds, int n_samples)

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

        self.act = np.zeros(in_nodes, dtype=np.float32, order='C')
        self._net_layer.act = <float *>np.PyArray_DATA(self.act)

        self.delta = np.zeros(in_nodes, dtype=np.float32, order='C')
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

    def train(self, features, labels, max_iter, alpha, lamb, decay):
        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert features.shape[0] == labels.shape[0]
        assert features.shape[1] == self.n_features
        assert labels.shape[1] == self.n_labels

        for i in xrange(max_iter):
            start = time.time()
            # shuffle data set
            idx = np.random.permutation(features.shape[0])
            _features = features[idx].copy(order='C').astype(np.float32)
            _labels = labels[idx].copy(order='C').astype(np.float32)
            print "data set shuffled"
            neural_sgd_iteration(self.head.get_ptr(), self.tail.get_ptr(), 
                    <float *>np.PyArray_DATA(_features), <float *>np.PyArray_DATA(_labels), 
                    features.shape[0], alpha, lamb)
            end = time.time()
            print "iteration %d completed in %f seconds" % (i, end - start)
            alpha *= decay

    def predict_prob(self, features):
        assert isinstance(features, np.ndarray)
        assert features.shape[1] == self.n_features

        _features = features.copy(order='C').astype(np.float32)
        preds = np.zeros((features.shape[0], self.n_labels), dtype=np.float32, order='C')
        neural_predict_prob(self.head.get_ptr(), self.tail.get_ptr(),
                <float *>np.PyArray_DATA(_features), <float *>np.PyArray_DATA(preds), 
                features.shape[0])
        return preds

    def predict_label(self, features):
        preds = self.predict_prob(features)
        return np.argmax(preds, axis=1)
