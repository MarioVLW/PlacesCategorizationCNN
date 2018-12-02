import tensorflow as tf
import math

class NetworkCNN:

    def __init__(self, DICT_CNN_PARAMS):
        self.DICT_CNN_PARAMS = DICT_CNN_PARAMS
        self.training_mode = True
        self.weights = {}
        self.biases = {}

    def weight_biases_creator(self):
        conv_number = self.DICT_CNN_PARAMS["conv_number"]
        for conv in range(conv_number):
            if 0 == conv:
                prev_dimension = self.DICT_CNN_PARAMS["image_dim"]
            else:
                prev_dimension = self.DICT_CNN_PARAMS["conv_dimensions"][conv-1]
            actual_dim = self.DICT_CNN_PARAMS["conv_dimensions"][conv]
            kernel_size = self.DICT_CNN_PARAMS["conv_kernels"][conv]

            weight_name = "wc" + str(conv)
            variable_name = "W" + str(conv)

            self.weights.update({
                weight_name: tf.get_variable(variable_name, shape=(kernel_size, kernel_size, prev_dimension, actual_dim), initializer=tf.contrib.layers.xavier_initializer())
            })

            bias_name = "bc" + str(conv)
            variable_name = "B" + str(conv)

            self.biases.update({
                bias_name: tf.get_variable(variable_name, shape=actual_dim, initializer=tf.contrib.layers.xavier_initializer())
            })

        dense_number = self.DICT_CNN_PARAMS["dense_number"]
        for dense in range(dense_number):
            if 0 == dense:
                prev_dimension = math.ceil(self.DICT_CNN_PARAMS["image_size"]/pow(self.DICT_CNN_PARAMS["max_pooling_kernel"], conv_number))
                prev_dimension = pow(prev_dimension, 2)
                prev_dimension = prev_dimension * self.DICT_CNN_PARAMS["conv_dimensions"][conv_number-1]
            else:
                prev_dimension = self.DICT_CNN_PARAMS["dense_dimensions"][dense-1]

            actual_dim = self.DICT_CNN_PARAMS["dense_dimensions"][dense]
            variable_name = "W" + str(dense + conv_number)
            weight_name = "wd" + str(dense)

            self.weights.update({
                weight_name: tf.get_variable(variable_name, shape=(prev_dimension, actual_dim), initializer=tf.contrib.layers.xavier_initializer())
            })

            bias_name = "bd" + str(dense)
            variable_name = "B" + str(dense + conv_number)

            self.biases.update({
                bias_name: tf.get_variable(variable_name, shape=actual_dim, initializer=tf.contrib.layers.xavier_initializer())
            })

        variable_name = "W" + str(conv_number + dense_number)
        prev_dimension = self.DICT_CNN_PARAMS["dense_dimensions"][dense_number-1]

        self.weights.update({
            "out": tf.get_variable(variable_name, shape=(prev_dimension, self.DICT_CNN_PARAMS["number_classes"]), initializer=tf.contrib.layers.xavier_initializer())
        })

        variable_name = "B" + str(dense_number + conv_number)
        self.biases.update({
            "out": tf.get_variable(variable_name, shape=self.DICT_CNN_PARAMS["number_classes"], initializer=tf.contrib.layers.xavier_initializer())
        })

    def conv2d(self, x, w, b, strides=1, training_mode=True):
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        #x = tf.layers.batch_normalization(x, training=training_mode)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def predict_output(self, input_placeholder):
        conv_number = self.DICT_CNN_PARAMS["conv_number"]
        for conv in range(conv_number):
            weight_name = "wc" + str(conv)
            bias_name = "bc" + str(conv)
            if 0 == conv:
                conv_output = self.conv2d(input_placeholder, self.weights[weight_name], self.biases[bias_name], self.training_mode)
            else:
                conv_output = self.conv2d(conv_output, self.weights[weight_name], self.biases[bias_name], self.training_mode)

            conv_output = self.maxpool2d(conv_output, k=self.DICT_CNN_PARAMS["max_pooling_kernel"])

        dense_number = self.DICT_CNN_PARAMS["dense_number"]
        for dense in range(dense_number):
            weight_name = "wd" + str(dense)
            bias_name = "bd" + str(dense)
            if 0 == dense:
                full_conn_output = tf.reshape(conv_output, [-1, self.weights[weight_name].get_shape().as_list()[0]])

            full_conn_output = tf.add(tf.matmul(full_conn_output, self.weights[weight_name]), self.biases[bias_name])
            full_conn_output = tf.nn.relu(full_conn_output)

        predicted_output = tf.add(tf.matmul(full_conn_output, self.weights['out']), self.biases['out'])
        return predicted_output

    def set_training_algorithms(self):
        self.input_placeholder = tf.placeholder("float", [None, self.DICT_CNN_PARAMS["image_size"], self.DICT_CNN_PARAMS["image_size"], self.DICT_CNN_PARAMS["image_dim"]])
        self.output_placeholder = tf.placeholder("float", [None, self.DICT_CNN_PARAMS["number_classes"]])
        self.pred = self.predict_output(self.input_placeholder)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred, labels=self.output_placeholder))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.DICT_CNN_PARAMS["learning_rate"]).minimize(self.cost)
        # Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
        self.correct_prediction = tf.equal(tf.argmax(self.predict_output(self.input_placeholder), 1), tf.argmax(self.output_placeholder, 1))
        # calculate accuracy across all the given images and average them out.
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train_network(self, sess, batch):
        [batch_images, batch_labels] = batch
        sess.run(self.optimizer, feed_dict={self.input_placeholder: batch_images, self.output_placeholder: batch_labels})

    def update_indicators(self, sess, batch):
        [batch_images, batch_labels] = batch
        sess.run(self.optimizer, feed_dict={self.input_placeholder: batch_images, self.output_placeholder: batch_labels})
        loss, acc, pred = sess.run([self.cost, self.accuracy, self.pred], feed_dict={self.input_placeholder: batch_images, self.output_placeholder: batch_labels})
        return loss, acc, pred
