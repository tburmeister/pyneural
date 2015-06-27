PyNeural
========

A simple but fast Python library for training neural networks
-------------------------------------------------------------

### What is PyNeural?

PyNeural is a neural network library written in Cython which is powered by a
simple but fast C library under the hood. PyNeural uses the cblas library to
perform the backprogation algorithm efficiently on multicore processors.
PyNeural exposes a simple Python API that plays nicely with NumPy, making it
easy for you to munge your data sets as needed and quickly use them to train
a neural network for classifiction.

### Installation

0. Make sure you have the cblas library installed, or Xcode if on a Mac. If using a Mac, I recommend using Xcode's Accelerate framework, which includes a BLAS implementation. Otherwise, check out [OpenBLAS](https://github.com/xianyi/OpenBLAS).
1. Clone the repo.
2. If using OpenBLAS, you may need to edit `include_dirs` and `library_dirs` in `setup.py` to reflect your install location - it comes set to the default install location for OpenBLAS.
3. `python setup.py build_ext`  
`python setup.py install`  
(Note: you may need `sudo` for the install part.)

### Usage Example

Import the library:  
`import pyneural`

Initialize a neural net with 784 input features, 10 output classes, and 1
intermediate layer of 400 nodes:  
`nn = pyneural.NeuralNet([784, 400, 10])`

Train the network over 5 iterations of the training set with a mini-batch size
of 100, an alpha (gradient descent coefficient) of 0.01, an L2 penalty of 0.0,
and a decay multiplier of 1.0 (meaning alpha does not decrease with each
iteration):  
`nn.train(features, labels, 5, 100, 0.01, 0.0, 1.0)`

Get the predicted classes/labels:  
`prediction = nn.predict_label(features)`

Get the predicted probability of each class/label:  
`predicted_prob = nn.predict_prob(features)`

### Performance

On my Core i5 MacBook Pro, PyNeural can perform 5 iterations over the Kaggle
digits training set in approximately 45 seconds with a mini-batch size of 1, or
11 seconds with a mini-batch size of 200, versus 190 seconds for 5 iterations
with OpenCV, a roughly 4x to 17x speed-up.

For an example of the code that I used to achieve 99% accuracy on the Kaggle digit recognizer challenge with a training time of roughly 30 minutes, see `kaggle_digit_example.ipynb`.

### Full Example

The code below trains a neural net on the MNIST digits data set.
The neural net trained below has 784 input features, 10 output labels, and one
intermediate layer of 400 nodes. I have used a variation of this code to
achieve over 98% accuracy on the Kaggle test set for the digits recognizer
challenge.


    import pyneural
    import numpy as np
    import pandas as pd

    training_set = pd.read_csv('train.csv')
    labels = np.array(training_set)[:, 0]
    features = np.array(training_set)[:, 1:].astype(float) / 256

    n_rows = features.shape[0]
    n_features = features.shape[1]
    n_labels = 10
    
    labels_expanded = np.zeros((n_rows, n_labels))
    for i in xrange(n_rows):
        labels_expanded[i][labels[i]] = 1

    nn = pyneural.NeuralNet([n_features, 400, n_labels])
    nn.train(features, labels_expanded, 5, 1, 0.01, 0.0, 1.0)

    data set shuffled
    iteration 0 completed in 9.897468 seconds
    data set shuffled
    iteration 1 completed in 9.881156 seconds
    data set shuffled
    iteration 2 completed in 9.771729 seconds
    data set shuffled
    iteration 3 completed in 9.732244 seconds
    data set shuffled
    iteration 4 completed in 9.771301 seconds

    preds = nn.predict_label(features)
    n_correct = np.sum(preds == labels)
 
    print "%f%% percent correct on training set" % (100.0 * n_correct / n_rows)

    97.040476% percent correct 


### A Note on Mini-Batch size

Using mini-batches can significantly speed up the computation time of each iteration over the training set, but I have observed that increasing the mini-batch size only increases speed up to a point. For example, in the example above, (at least on my laptop) a mini-batch size of 100 is faster than a mini-batch size of 50, but a mini-batch size of 500 is slower. Further, the code uses slightly different computations for mini-batches of size 1, so a mini-batch of size 1 will be faster than a mini-batch of some small size greater than 1.

Additionally, increasing the size of the mini-batch decreases the rate of convergence ([source](http://www.cs.cmu.edu/~muli/file/minibatch_sgd.pdf)), so even though increasing the size of the mini-batch may increase the speed of each iteration, it also decreases the relative utility of that iteration. I'm not really sure what the best balance is, but anecdotally a mini-batch of size 100 seems to be a good default for this library.

### Word of Warning

This is still very much a work in progress. I can't guarantee it will work on
every platform, I'm sure there are plenty of ways for you to break it, and if
you do break it, it probably won't give you helpful error messages. I will try
to remedy all of these things, as well as add features and performance, in the
future.
