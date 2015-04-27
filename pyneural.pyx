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
            float *features, float *labels, const int n_samples, const int batch_size,
            const float alpha, const float lamb)

    void neural_predict_prob(neural_net_layer *head, neural_net_layer *tail,
            float *features, float *preds, int n_samples, int batch_size)

cdef class NetLayer:
    cdef neural_net_layer _net_layer
    cdef NetLayer prev, next
    cdef np.ndarray bias, theta, act, delta
    cdef int in_nodes, out_nodes

    def __init__(self, in_nodes, out_nodes):
        bias = 0.24 * np.random.rand(out_nodes) - 0.12
        self.bias = bias.copy(order='C').astype(np.float32)
        self._net_layer.bias = <float *>np.PyArray_DATA(self.bias)

        theta = 0.24 * np.random.rand(out_nodes * in_nodes) - 0.12
        self.theta = theta.copy(order='C').astype(np.float32)
        self._net_layer.theta = <float *>np.PyArray_DATA(self.theta)

        self._net_layer.in_nodes = self.in_nodes = in_nodes
        self._net_layer.out_nodes = self.out_nodes = out_nodes

    def set_batch(self, batch_size):
        self.act = np.zeros(self.in_nodes * batch_size, dtype=np.float32, order='C')
        self._net_layer.act = <float *>np.PyArray_DATA(self.act)

        self.delta = np.zeros(self.in_nodes * batch_size, dtype=np.float32, order='C')
        self._net_layer.delta = <float *>np.PyArray_DATA(self.delta)

    cdef void set_prev(self, NetLayer prev):
        self.prev = prev
        self._net_layer.prev = &prev._net_layer

    cdef void set_next(self, NetLayer next):
        self.next = next
        self._net_layer.next = &next._net_layer

    cdef neural_net_layer *get_ptr(self):
        return &self._net_layer

cdef class NeuralNet:
    cdef list layers
    cdef int n_features, n_labels
    cdef NetLayer head, tail

    def __init__(self, layers):
        """
        Initialize a new neural network object.

        Args:
            layers (list of integers): The number of nodes at each layer of the
                network. The first value should be the number of input features
                and the last value should be the number of output labels;
                intermediate values, if any, are the number of nodes in the
                hidden layers.

        Returns:
            class NeuralNet: a new neural network object, with model parameters
                randomly initialized using self.random_init().

        Example: 
            Create a new neural network with 784 input features, 10 output
            layers, and one hidden layer of 400 nodes.

            >>> nn = pyneural.NeuralNet([784, 400, 10])

        """
        assert len(layers) >= 2
        self.layers = list(layers)
        self.n_features = self.layers[0]
        self.n_labels = self.layers[-1]
        self.head = None
        self.tail = None
        self.random_init()

    def random_init(self):
        """
        Randomly initialize the model parameters of the neural network.

        """
        self.head = NetLayer(self.layers[0], self.layers[1])
        prev = self.head

        for k in xrange(1, len(self.layers) - 1):
            curr = NetLayer(self.layers[k], self.layers[k + 1])
            curr.set_prev(prev)
            prev.set_next(curr)
            prev = curr

        self.tail = NetLayer(self.layers[-1], 0)
        self.tail.set_prev(prev)
        prev.set_next(self.tail)

    def train(self, features, labels, max_iter, batch_size, alpha, lamb, decay):
        """
        Train the neural network on a training set with known output labels.

        Args:
            features (2-dim'l numpy.ndarray): A row for each training example
                consisting of that examples features.
            labels (2-dim'l numpy.ndarray): A row for each training example and
                column for each output label, denoting whether the example
                belongs to that class (1) or not (0).
            max_iter (int): The number of times to iterate over the training set.
            batch_size (int): The mini-batch size.
            alpha (float): The gradient descent learning rate.
            lamb (float): The L2 penalty coefficient.
            decay (float): The amount to multiply the learning rate by after
                each iteration over the training set.

        Example:
            Train a neural network on a small training set learning rate 0.1,
            no L2 penalty, and no learning rate decay over 10 iterations, with
            a mini-batch size of 100.

            >>> features = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
            array([[1, 0, 1, 0],
                   [0, 1, 0, 1]])

            >>> labels = np.array([[1, 0], [0, 1]])
            array([[1, 0],
                   [0, 1]])

            >>> nn.train(features, labels, 10, 100, 0.1, 0.0, 1.0)

        """
        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert features.shape[0] == labels.shape[0]
        assert features.shape[1] == self.n_features
        assert labels.shape[1] == self.n_labels

        # allocate space based on batch size
        curr = self.head
        while curr != None:
            curr.set_batch(batch_size)
            curr = curr.next

        for i in xrange(max_iter):
            start = time.time()
            # shuffle data set
            idx = np.random.permutation(features.shape[0])
            _features = features[idx].copy(order='C').astype(np.float32)
            _labels = labels[idx].copy(order='C').astype(np.float32)
            print "data set shuffled"
            neural_sgd_iteration(self.head.get_ptr(), self.tail.get_ptr(), 
                    <float *>np.PyArray_DATA(_features), <float *>np.PyArray_DATA(_labels), 
                    features.shape[0], batch_size, alpha, lamb)
            end = time.time()
            print "iteration %d completed in %f seconds" % (i, end - start)
            alpha *= decay

    def predict_prob(self, features, batch_size):
        """
        Given a set of example features, predict the probabilities of each
        example belonging to each label.

        Args:
            features (2-dim'l numpy.ndarray): A row for each example consisting
                of its features.

        Returns:
            2-dim'l numpy.ndarray: The probabilities of each example belonging
                to each output label.

        """
        assert isinstance(features, np.ndarray)
        assert features.shape[1] == self.n_features

        _features = features.copy(order='C').astype(np.float32)
        preds = np.zeros((features.shape[0], self.n_labels), dtype=np.float32, order='C')
        neural_predict_prob(self.head.get_ptr(), self.tail.get_ptr(),
                <float *>np.PyArray_DATA(_features), <float *>np.PyArray_DATA(preds), 
                features.shape[0], batch_size)
        return preds

    def predict_label(self, features, batch_size):
        """
        Given a set of example features, predict the most probable output label
        for each example.

        Args:
            features (2-dim'l numpy.ndarray): A row for each example consisting
                of its features.

        Returns:
            1-dim'l numpy.ndarray: The most probable label for each example.

        """
        preds = self.predict_prob(features, batch_size)
        return np.argmax(preds, axis=1)
