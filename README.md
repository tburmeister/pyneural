PyNeural
========

A simple but fast Python library for training neural networks
-------------------------------------------------------------

### What is PyNeural?

PyNeural is a neural network library written in Cython which is powered by a
simple but fast C library under the hood. PyNeural uses the cblas library to
perform the backprogation algorithm efficiently on multicore processors.
PyNeural exposes a simple Python API that plays nicely with NumPy, making it
easy to for you to munge your data sets as needed and quickly use them to train
a neural network for classifiction.

### Installation

0. Make sure you have the cblas library installed, or Xcode if on a Mac.
1. Clone the repo.
2. If on a Mac:
`CC=clang CFLAGS="-DACCELERATE -framework accelerate" python setup.py build_ext -i`  
If on Linux:
`CC=clang python setup.py build_ext -i`  
(Note: I realize this is hacky - will add a proper makefile or install script
later.)

### Use

Import the library:  
`import pyneural`

Initialize a neural net with 784 input features, 10 output classes, and 1
intermediate layer of 400 nodes:  
`nn = pyneural.NeuralNet(784, 10, 400, 1)`

Train the network over 5 iterations of the training set with an alpha (gradient
descent coefficient) of 0.01, an L2 penalty of 0.0, and a decay multiplier of
1.0 (meaning alpha does not decrease with each iteration):  
`nn.train(features, labels, 5, 0.01, 0.0, 1.0)`

Get the predicted classes/labels:  
`prediction = nn.predict_label(features)`

Get the predicted probability of each class/label:  
`predicted_prob = nn.predict_prob(features)`

### Performance

On my Core i5 MacBook Pro, PyNeural can perform 5 iterations over the Kaggle digits training set in approximately 45 seconds, versus 190 seconds for 5 iterations with OpenCV, a roughly 4x speed-up. 

### Example

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

    nn = pyneural.NeuralNet(n_features, n_labels, 400, 1)
    nn.train(features, labels_expanded, 5, 0.01, 0.0, 1.0)

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
    
    correct = 0
    for i in xrange(n_rows):
        if preds[i] == labels[i]:
            correct += 1
            
    print "%f%% percent correct on training set" % (100.0 * correct / n_rows)

    97.040476% percent correct 


### Word of Warning

This is still very much a work in progress. I can't guarantee it will work on
every platform, I'm sure there are plenty of ways for you to break it, and if
you do break it, it probably won't give you helpful error messages. I will try
to remedy all of these things, as well as add features and performance, in the
future. I will also try to add actual documentation to the code and docstrings to
the functions.
