import numpy as np
cimport numpy as np

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

cdef neural_net_layer *_random_layer(int in_nodes, int out_nodes):
    cdef neural_net_layer layer
    # TODO: add actual values
    layer.in_nodes = in_nodes
    layer.out_nodes = out_nodes
    return &layer

cdef class NeuralNet:
    cdef neural_net_layer *head
    cdef neural_net_layer *tail
    cdef int features, labels, nodes, layers

    def __init__(self, features, labels, nodes, layers):
        self.features = features
        self.labels = labels
        self.nodes = nodes
        self.layers = layers
        self.random_init()

    def random_init(self):
        self.head = _random_layer(self.features, self.nodes)
        cdef neural_net_layer *prev = self.head
        cdef neural_net_layer *curr

        for k in xrange(self.layers - 1):
            curr = _random_layer(self.nodes, self.nodes)
            curr.prev = prev
            prev.next = curr
            prev = curr

        curr = _random_layer(self.nodes, self.labels)
        curr.prev = prev
        prev.next = curr
        prev = curr

        self.tail = _random_layer(self.labels, 0)
        self.tail.prev = prev

    def _feed_forward(self, np.ndarray[float, ndim=2, mode="c"] x not None):
        neural_feed_forward(self.head, <float *>np.PyArray_DATA(x))

    def _back_prop(self, np.ndarray[float, ndim=2, mode="c"] y not None, alpha, lamb):
        neural_back_prop(self.tail, <float *>np.PyArray_DATA(y), alpha, lamb)

    def _sgd_iteration(self, np.ndarray[float, ndim=2, mode="c"] features not None, 
                       np.ndarray[float, ndim=2, mode="c"] labels not None, 
                       alpha, lamb):
        neural_sgd_iteration(self.head, self.tail, 
                <float *>np.PyArray_DATA(features), <float *>np.PyArray_DATA(labels), 
                features.shape[0], alpha, lamb)
