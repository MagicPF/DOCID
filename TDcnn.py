"""
"  @title: cnn.py
"  @version: 4/19/17
"""

# imports
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
import os
import sys

# constants and globals
start_time = time.time()
x = tf.placeholder('float')
y = tf.placeholder('float')
img_dimension = 50
num_slices = 20
num_classes = 2

# load in processed data from the results of preprocess.py
processed_data = []
# could probably work on these values for the training/validation set to better train the CNN for other data sets
training_set = processed_data[:-100]
validation_set = processed_data[-100:]

# reference: https://www.tensorflow.org/tutorials/deep_cnn
def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

# reference: https://www.tensorflow.org/tutorials/deep_cnn
def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

# reference (see lines 6-12): https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/
def cnn(x):
    # had better accuracies with 3,3,3 rather than 5,5,5
    # 32 features in convolution 1, 64 features in convolution 2
    # setup weights for convolution 1 and 2
    weights = {'convolution1_weights':tf.Variable(tf.random_normal([3,3,3,1,32])), 'convolution2_weights':tf.Variable(tf.random_normal([5,5,5,32,64])),
               'fc_weights':tf.Variable(tf.random_normal([54080,1024])), 'out':tf.Variable(tf.random_normal([1024, num_classes]))}

    # setup biases for convolution 1 and 2
    biases = {'convolution1_biases':tf.Variable(tf.random_normal([32])), 'convolution2_biases':tf.Variable(tf.random_normal([64])),
              'fc_biases':tf.Variable(tf.random_normal([1024])), 'out':tf.Variable(tf.random_normal([num_classes]))}

    # reshape data, this easily throws errors if you dont do preprocessing correctly
    x = tf.reshape(x, shape=[-1, img_dimension, img_dimension, num_slices, 1]) #第一个原本是-1

    # convolution 1 with max pooling
    convolution1 = tf.nn.relu(conv3d(x, weights['convolution1_weights']) + biases['convolution1_biases'])
    convolution1 = maxpool3d(convolution1)

    # convolution 2 with max pooling
    convolution2 = tf.nn.relu(conv3d(convolution1, weights['convolution2_weights']) + biases['convolution2_biases'])
    convolution2 = maxpool3d(convolution2)

    fc = tf.reshape(convolution2,[-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['fc_weights'])+biases['fc_biases'])
    fc = tf.nn.dropout(fc, 0.8)

    output = tf.matmul(fc, weights['out'])+biases['out']
    return output

# reference (see lines 6-12): https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/
def training(x,flag):
    losses = []
    new_time = time.time()
    if flag == 1:
        training_set = np.load('processed_Tdata.npy')
    else:
        training_set = np.load('processed_Rdata.npy') 
    
    validation_set = np.load('processed_Rdata.npy') 
    prediction = cnn(x)
    # prediction = tf.expand_dims(prediction, axis=2)
    print("prediction :",prediction)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # can play wtih the learning rate here to achieve different results, originally it was .0001
    optimizer = tf.train.AdamOptimizer(learning_rate=.01).minimize(cost)

    # 10 epochs is enough to get close to an optimum accuracy, model usually reaches an upper bound by 14-15 epochs
    # not sure which is better, I think we may be overfitting the data with 20 but the accuracy is MUCH better by that point
    num_epochs = 20
    # setup our tensorflow session
    # also used reference: https://www.tensorflow.org/tutorials/deep_cnn
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # setup our counts
        success_count = 0
        runs_count = 0
        # run for each epoch
        for epoch in range(num_epochs):
            # timing is pretty nice here since this stuff can take FOREVER
            loss = 0
            new_time = time.time()
            # print("training set is :",training_set)
            for data in training_set:
                runs_count += 1
                # print("data is :",data)
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    loss += c
                    success_count += 1
                    print("X:",X)
                    print("Y:",Y)
                    print("Cost: ",cost)
                    print("optimizer: ",optimizer)
                    print("C:",c)
                    print("loss: ",loss)
                    losses.append(loss)
                    # a = input();
                    
                except Exception as e:
                    print(str(e))
                    pass
            
            print("Completed epoch " + str(epoch+1) + " out of " + str(num_epochs))
            print("\tLoss: " + str(loss))
            print("\tTime to complete: %s seconds." % (time.time() - new_time))
            print("prediction arg:",prediction)
            print("y arg:",y)
            correct = tf.equal(tf.argmax(prediction, 0), tf.argmax(y, 0))
            print("correct",correct)
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print("accuracy = ",accuracy)
            # accuracy_local = accuracy.eval({x:[i[0] for i in validation_set], y:[i[1] for i in validation_set]})
            # print('\tAccuracy: ' + str(accuracy_local) + "\n")
            
        print("Jobs Done!")
        print("Val is ",validation_set)
        print("Accuracy is ",accuracy)
        accuracy = accuracy.eval({x: [i[0] for i in validation_set], y: [i[1] for i in validation_set]})
        print("runs_count = ",runs_count)
        fitment = success_count/runs_count
        print("Finishing accuracy: " + str(accuracy))
        print("Finishing fitment: " + str(fitment))
        print("Total running time: %s seconds." % (time.time() - new_time))

        # write (append) results to output.txt
        fh = open("output.txt", "a+")
        fh.write(str(accuracy) + "\n")
        fh.close()
        return accuracy,losses
def trainR():
    processed_data = np.load('processed_Rdata.npy')
    # could probably work on these values for the training/validation set to better train the CNN for other data sets
    training_set = processed_data
    validation_set = [1,0]
    fh = open("output.txt", "a+")
    fh.write("=== Beginning new Trial ===\n")
    fh.close()
    print("Beginning first cnn interation...")
    accuracy,loss = training(x,0)
    while(accuracy < 0.69):
        print("\n === Accuracy is less than target, trying a new iteration... ===")
        accuracy = training(x,1)
        # os.system("echo ")
    print("Success! with C loss ",loss)
    print("It took %s seconds to achieve an accuracy >= 0.69" % (time.time() - start_time))
def trainT():
    processed_data = np.load('processed_Tdata.npy')
    # could probably work on these values for the training/validation set to better train the CNN for other data sets
    training_set = processed_data
    validation_set = [0,1]
    fh = open("output.txt", "a+")
    fh.write("=== Beginning new Trial ===\n")
    fh.close()
    print("Beginning first cnn interation...")
    accuracy,loss = training(x,0)
    while(accuracy < 0.69):
        print("\n === Accuracy is less than target, trying a new iteration... ===")
        accuracy = training(x,1)
    print("Success! with D loss ",loss)
    print("It took %s seconds to achieve an accuracy >= 0.69" % (time.time() - start_time))
  
