from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import random
import sys

from . import utils


def train(x_train, y_train, x_test, y_test, epochs=100, batch_size=128, a=0.1, prox_const=0.00001):
    """Training function, better description to come

    Arguments:
        x_train: ndarray (samplesxfeatures)
            The training set
        y_train: ndarray (samplesx1)
            The labels of the training set
        x_test: ndarray (samplesxfeatures)
            The test set
        y_test: ndarray (samplesx1)
            The labels of the test set
        epochs: int, default 100
            Number of training iterations
        batch_size: int, default 128
            Number of samples to process at a time in each epoch
        a: float, default ______
            Gradient descent change parameter
        prox_const: float, default ______
            Threshold value for soft thresholding
    
    Returns
        w: ndarray (featuresx1)
        b: float
    """
    n_samples, n_features = x_train.shape
    
    w = tf.Variable(np.random.rand(n_features, 1).astype(dtype='float64'), name="w")
    b = tf.Variable(0.0, dtype=tf.float64, name="b")

    x = tf.placeholder(dtype=tf.float64, name='x')
    y = tf.placeholder(dtype=tf.float64, name='y')

    predictions = tf.matmul(x, w) + b
    loss = tf.reduce_mean(
        tf.log(1 + tf.exp(
            tf.multiply(-1.0*y, predictions)
        ))
    )

    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(epochs):
            for i in range(0,n_samples,batch_size):
                iE = min(n_samples, i+batch_size)
                x_batch = x_train[i:iE,:]
                y_batch = y_train[i:iE,:]
                sess.run([train],feed_dict={x: x_batch, y: y_batch})
            # training done in this epoch
            # but, just so that the user can monitor progress, try current w,b on full test set
            y_pred,curr_loss,curr_w,curr_b=sess.run([predictions,loss,w,b],feed_dict={x: x_test, y: y_test})
            MSE=np.mean(np.mean(np.square(y_pred-y_test),axis=1),axis=0)

            # Soft thresholding
            for i in range(len(curr_w)):
                if curr_w[i][0] < prox_const*-1:
                    curr_w[i][0] += prox_const
                elif curr_w[i][0] > prox_const:
                    curr_w[i][0] -= prox_const
                else:
                    curr_w[i][0] = 0
            sess.run([tf.assign(w, curr_w)])

            print("Loss: {:.3f}".format(curr_loss))
            print("MSE: {:.3f}".format(MSE))

    return curr_w, curr_b


def predict(w, b, test):
    for item in test:
        item = np.atleast_2d(item)
        u = np.matmul(item,w) + b
        if u < 0:
            yield -1
        elif u > 0:
            yield 1


def main(argv):
    C0 = 0
    C1 = 8

    # Read args from command line
    sampleSize = utils.parseArgs(argv)

    # Load the train and test sets from MNIST
    print("Loading datasets from MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Apply preprocessing to the training and test sets
    print("Preprocessing training set...")
    x_train, y_train = utils.preprocess(x_train, y_train, C0, C1)
    print("Preprocessing testing set...")
    x_test, y_test = utils.preprocess(x_test, y_test, C0, C1)

    # Sample training set
    sampleIndicies = random.sample(range(len(x_train)), int(len(x_train)*sampleSize))
    x_train_sample = np.array([_ for i, _ in enumerate(x_train) if i in sampleIndicies])
    y_train_sample = np.array([_ for i, _ in enumerate(y_train) if i in sampleIndicies])

    print("Training model...")
    w, b = train(x_train_sample, y_train_sample, x_test, y_test)

    print("Evaluating model...")
    labels = predict(w, b, x_test)

    print("Calculating metrics...")
    utils.evaluate(labels, y_test)

if __name__ == '__main__':
    main(sys.argv)
