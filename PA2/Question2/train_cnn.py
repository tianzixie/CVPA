import time


import tensorflow as tf

########### Convolutional neural network class ############
class ConvNet(object):
    def __init__(self, mode):
        self.mode = mode

    # Read train, valid and test data.
    def read_data(self, train_set, test_set):
        # Load train set.
        trainX = train_set.images
        trainY = train_set.labels

        # Load test set.
        testX = test_set.images
        testY = test_set.labels

        return trainX, trainY, testX, testY

    # Baseline model. step 1
    def model_1(self, X, hidden_size):
        X_ = tf.contrib.layers.flatten(X)
        num_output = 10
        fully_connected_layer = tf.layers.dense(X_, hidden_size, activation=tf.nn.sigmoid)
        logits = tf.layers.dense(fully_connected_layer, num_output)
        return logits


    # Use two convolutional layers.
    def model_2(self, X, hidden_size):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer. and pooling layer
        #
        # ----------------- YOUR CODE HERE ----------------------
        # first conv + pool
        convLayer1 = tf.layers.conv2d(X, filters=20, kernel_size=5, 
                                      padding="VALID", activation=tf.nn.sigmoid)
        poolLayer1 = tf.layers.max_pooling2d(convLayer1, pool_size=2, 
                                             strides=2, padding="VALID")

        # second conv + pool
        convLayer2 = tf.layers.conv2d(poolLayer1, filters=40, kernel_size=5, 
                                      padding="VALID", activation=tf.nn.sigmoid)
        poolLayer2 = tf.layers.max_pooling2d(convLayer2, pool_size=2, 
                                             strides=2, padding="VALID")

        # fully connected
        poolLayer2 = tf.contrib.layers.flatten(poolLayer2)
        num_output = 10
        fullyConLayer = tf.layers.dense(poolLayer2, hidden_size, activation=tf.nn.sigmoid)
        logits = tf.layers.dense(fullyConLayer, num_output)
        return logits

    # Replace sigmoid with ReLU.
    def model_3(self, X, hidden_size):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        # first conv + pool
        convLayer1 = tf.layers.conv2d(X, filters=20, kernel_size=5, 
                                      padding="VALID", activation=tf.nn.relu)
        poolLayer1 = tf.layers.max_pooling2d(convLayer1, pool_size=2, 
                                             strides=2, padding="VALID")

        # second conv + pool
        convLayer2 = tf.layers.conv2d(poolLayer1, filters=40, kernel_size=5, 
                                      padding="VALID", activation=tf.nn.relu)
        poolLayer2 = tf.layers.max_pooling2d(convLayer2, pool_size=2, 
                                             strides=2, padding="VALID")

        # fully connected
        poolLayer2 = tf.contrib.layers.flatten(poolLayer2)
        num_output = 10
        fullyConLayer = tf.layers.dense(poolLayer2, hidden_size, activation=tf.nn.sigmoid)
        logits = tf.layers.dense(fullyConLayer, num_output)
        return logits

    # Add one extra fully connected layer.
    def model_4(self, X, hidden_size, decay):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        # first conv + pool
        convLayer1 = tf.layers.conv2d(X, filters=20, kernel_size=5, 
                                      padding="VALID", activation=tf.nn.relu)
        poolLayer1 = tf.layers.max_pooling2d(convLayer1, pool_size=2, 
                                             strides=2, padding="VALID")

        # second conv + pool
        convLayer2 = tf.layers.conv2d(poolLayer1, filters=40, kernel_size=5, 
                                      padding="VALID", activation=tf.nn.relu)
        poolLayer2 = tf.layers.max_pooling2d(convLayer2, pool_size=2, 
                                             strides=2, padding="VALID")

        # fully connected
        poolLayer2 = tf.contrib.layers.flatten(poolLayer2)
        num_output = 10
        fullyConLayer1 = tf.layers.dense(poolLayer2, hidden_size, activation=tf.nn.sigmoid)

        # fully connected with L2 regularizer
        l2Regularizer = tf.contrib.layers.l2_regularizer(decay)
        fullyConLayer2 = tf.layers.dense(fullyConLayer1, hidden_size, activation=tf.nn.sigmoid,
                                         kernel_regularizer=l2Regularizer)

        logits = tf.layers.dense(fullyConLayer2, num_output)
        return logits

    # Use Dropout now.
    def model_5(self, X, hidden_size, is_train):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        
        # first conv + pool
        convLayer1 = tf.layers.conv2d(X, filters=20, kernel_size=5, 
                                      padding="VALID", activation=tf.nn.relu)
        poolLayer1 = tf.layers.max_pooling2d(convLayer1, pool_size=2, 
                                             strides=2, padding="VALID")

        # second conv + pool
        convLayer2 = tf.layers.conv2d(poolLayer1, filters=40, kernel_size=5, 
                                      padding="VALID", activation=tf.nn.relu)
        poolLayer2 = tf.layers.max_pooling2d(convLayer2, pool_size=2, 
                                             strides=2, padding="VALID")

        # fully connected
        poolLayer2 = tf.contrib.layers.flatten(poolLayer2)
        num_output = 10
        fullyConLayer1 = tf.layers.dense(poolLayer2, hidden_size, activation=tf.nn.sigmoid)

        # fully connected with L2 regularizer
        l2Regularizer = tf.contrib.layers.l2_regularizer(decay)
        fullyConLayer2 = tf.layers.dense(fullyConLayer1, hidden_size, activation=tf.nn.sigmoid,
                                         kernel_regularizer=l2Regularizer)
        # dropout
        dropRate = 0.5
        dropout = tf.layers.dropout(fullyConLayer2, dropRate, training=is_train)

        logits = tf.layers.dense(dropout, num_output)
        return logits

    # Entry point for training and evaluation.
    def train_and_evaluate(self, FLAGS, train_set, test_set):
        class_num = 10
        num_epochs = FLAGS.num_epochs
        batch_size = FLAGS.batch_size
        learning_rate = FLAGS.learning_rate
        hidden_size = FLAGS.hiddenSize
        decay = FLAGS.decay

        trainX, trainY, testX, testY = self.read_data(train_set, test_set)
        
        input_size = trainX.shape[1]
        train_size = trainX.shape[0]
        test_size = testX.shape[0]

        trainX = trainX.reshape((-1, 28, 28, 1))
        testX = testX.reshape((-1, 28, 28, 1))

        with tf.Graph().as_default():
            # Input data
            X = tf.placeholder(tf.float32, [None, 28, 28, 1])
            Y = tf.placeholder(tf.int64, [None])
            is_train = tf.placeholder(tf.bool)

            # model 1: base line
            if self.mode == 1:
                features = self.model_1(X, hidden_size)

            # model 2: use two convolutional layer
            elif self.mode == 2:
                features = self.model_2(X, hidden_size)

            # model 3: replace sigmoid with relu
            elif self.mode == 3:
                features = self.model_3(X, hidden_size)

            # model 4: add one extral fully connected layer
            elif self.mode == 4:
                features = self.model_4(X, hidden_size, decay)

            # model 5: utilize dropout
            elif self.mode == 5:
                features = self.model_5(X, hidden_size, is_train)

            # ======================================================================
            # Define softmax layer, use the features.
            # ----------------- YOUR CODE HERE ----------------------
            logits = features
            # ======================================================================
            # Define loss function, use the logits.
            # ----------------- YOUR CODE HERE ----------------------
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)
            loss = tf.reduce_mean(cross_entropy)
            # ======================================================================
            # Define training op, use the loss.
            # ----------------- YOUR CODE HERE ----------------------
            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            # ======================================================================
            # Define accuracy op.
            # ----------------- YOUR CODE HERE ----------------------
            correct = tf.nn.in_top_k(logits, Y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            # ======================================================================
            # Allocate percentage of GPU memory to the session.
            # If you system does not have GPU, set has_GPU = False
            #
            has_GPU = False
            if has_GPU:
                gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
                config = tf.ConfigProto(gpu_options=gpu_option)
            else:
                config = tf.ConfigProto()

            # Create TensorFlow session with GPU setting.
            with tf.Session(config=config) as sess:
                tf.global_variables_initializer().run()

                for i in range(num_epochs):
                    print(20 * '*', 'epoch', i + 1, 20 * '*')
                    start_time = time.time()
                    s = 0
                    while s < train_size:
                        e = min(s + batch_size, train_size)
                        batch_x = trainX[s: e]
                        batch_y = trainY[s: e]
                        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_train: True})
                        s = e
                    end_time = time.time()
                    print ('the training took: %d(s)' % (end_time - start_time))

                    test_accuracy = accuracy.eval(feed_dict={X: testX, Y: testY, is_train: False})
                    #test_accuracy = sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False})
                    print ('accuracy of the trained model %f' % test_accuracy)
                    print ()

                return accuracy.eval(feed_dict={X: testX, Y: testY, is_train: False})





