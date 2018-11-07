from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import sys
import math

from utils import utils


# def f(w):
#     sum = 0
#     for item in w:
#         sum += item[0]
    
    


def train(x_train, y_train, x_test, y_test, epochs=85, batch_size=128, a=0.0001):
    n_samples, n_features = x_train.shape
    # print(x_train.shape, y_train.shape)
    # y_train = np.reshape(y_train, (len(y_train), 1))
    # y_train = np.atleast_2d(y_train)
    
    w = tf.Variable(np.random.rand(n_features, 1).astype(dtype='float64'), name="w")
    b = tf.Variable(0.0, dtype=tf.float64, name="b")

    x = tf.placeholder(dtype=tf.float64,name='x')
    y = tf.placeholder(dtype=tf.float64,name='y')

    predictions = tf.matmul(x, w) + b
    # loss = tf.log(1 + tf.exp(tf.matmul(tf.matmul(tf.negative(y), tf.transpose(w)), x)))
    loss = tf.matmul(tf.matmul(tf.negative(y), tf.transpose(w)), tf.transpose(x))
    risk = tf.reduce_mean(loss)

    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(risk)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # calculate and print Mean Squared Error on the full test set, using initial (random) w,b
        # y_pred=sess.run([predictions],feed_dict={x: x_test, y: y_test})[0];
        # MSE=np.mean(np.square(y_pred-y_test),axis=0);
        # print(MSE)

        for _ in range(epochs):
            for i in range(0,n_samples,batch_size):
                iE=min(n_samples,i+batch_size)
                x_batch=x_train[i:iE,:]
                y_batch=y_train[i:iE,:]
                _,curr_batch_risk,predBatchY=sess.run([train,risk,predictions],feed_dict={x: x_batch, y: y_batch})
            # training done in this epoch
            # but, just so that the user can monitor progress, try current w,b on full test set
            y_pred,curr_w,curr_b=sess.run([predictions,w,b],feed_dict={x: x_test, y: y_test})
            MSE1=np.mean(np.mean(np.square(y_pred-y_test),axis=1),axis=0)

            prox_const = 0.00001
            for i in range(len(curr_w)):
                if curr_w[i][0] < prox_const*-1:
                    curr_w[i][0] += prox_const
                elif curr_w[i][0] > prox_const:
                    curr_w[i][0] -= prox_const
                else:
                    curr_w[i][0] = 0
            sess.run([tf.assign(w, curr_w)])

            y_pred,_,_=sess.run([predictions,w,b],feed_dict={x: x_test, y: y_test})
            MSE2=np.mean(np.mean(np.square(y_pred-y_test),axis=1),axis=0)

            print("{:.2f}   {:.2f}".format(MSE1, MSE2))

    # print(np.transpose(curr_w))
    # print(curr_b)
    return curr_w, curr_b


def predict(w, b, test):
    labels = list()
    for item in test:
        item = np.atleast_2d(item)
        u = np.matmul(item,w) + b
        prob = int(1.0 / (1.0 + np.exp(np.negative(u))))
        if prob == 0:
            prob = -1
        labels.append(prob)
    return labels

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
    
    # Apply feature selection to training set
    # print("Applying feature selection...")
    # x_train, x_test = utils.featureSelection(x_train, x_test)
    
    x_test = x_test[:int(len(x_test)/3)]
    y_test = y_test[:int(len(y_test)/3)]

    print("Training model...")
    w, b = train(x_train, y_train, x_test, y_test)

    print("Evaluating model...")
    labels = predict(w, b, x_test)

    print("Calculating metrics...")
    utils.evaluate(labels, y_test)

if __name__ == '__main__':
    main(sys.argv[1:])
