# import depencies
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("./data", one_hot=True)

num_training = mnist_data.train.num_examples
num_testing = mnist_data.test.num_examples
num_validation = mnist_data.validation.num_examples
print("MNIST Dataset : Training sample: {0}, Testing sample : {1}, Validation sample : {2}".format(num_training, num_testing, num_validation))

n_input = 784
n_hidden_1 = 512
n_hidden_2 = 256
n_hidden_3 = 128
n_output = 10

learning_rate = 1e-4
epochs = 3000
batch_size = 128
keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])

nn_weight = {"W1": tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
             "W2": tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
             "W3": tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
             "Wout": tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))
             }
nn_bias = {"B1": tf.Variable(tf.truncated_normal([n_hidden_1])),
           "B2": tf.Variable(tf.truncated_normal([n_hidden_2])),
           "B3": tf.Variable(tf.truncated_normal([n_hidden_3])),
           "B4": tf.Variable(tf.truncated_normal([n_output])),
          }

nn_layer_1 = tf.add(tf.matmul(X,nn_weight["W1"]), nn_bias["B1"])
nn_layer_2 = tf.add(tf.matmul(nn_layer_1,nn_weight["W2"]), nn_bias["B2"])
nn_layer_3 = tf.add(tf.matmul(nn_layer_2,nn_weight["W3"]), nn_bias["B3"])
layer_drop = tf.nn.dropout(nn_layer_3, keep_prob)
output_layer = tf.add(tf.matmul(layer_drop,nn_weight["Wout"]), nn_bias["B4"])

computed_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = output_layer, labels = Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(computed_loss)
prediction_out = tf.equal(tf.arg_max(output_layer,1), tf.argmax(Y,1))
nn_accuracy = tf.reduce_mean(tf.cast(prediction_out, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        mini_batch_x, mini_batch_y = mnist_data.train.next_batch(batch_size)
        mini_batch_val_x, mini_batch_val_y = mnist_data.validation.next_batch(batch_size)

        sess.run(optimizer, feed_dict={X:mini_batch_x,Y:mini_batch_y, keep_prob:1})
        if 1%100 == 0:
            mini_batch_loss, mini_batch_accuracy = sess.run([computed_loss,nn_accuracy], feed_dict={X:mini_batch_x,Y:mini_batch_y, keep_prob:1})
            mini_batch_val_loss, mini_batch_val_accuracy = sess.run([computed_loss, nn_accuracy],feed_dict={X: mini_batch_val_x, Y: mini_batch_val_y, keep_prob: 1})
            print("Iteration : {0}, Train_loss = {1}, Train_accuracy {2} Val_loss {3}, Val_accuracy {4}".format(i,mini_batch_loss, mini_batch_accuracy, mini_batch_val_loss, mini_batch_val_accuracy))
    print("Optimizer Finished")
    test_accuracy = sess.run(nn_accuracy, feed_dict={X:mnist_data.test.images, Y:mnist_data.test.labels, keep_prob:1.0})
    print("testing accuracy is {0}".format(test_accuracy))

    saver_path = saver.save(sess, "./model/my_model.ckpt")

img = cv2.imread("7.jpeg")
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rescaled_image = cv2.resize(gray_image,(28,28))
plt.imshow(rescaled_image,cmap='gray')
plt.show()
rescaled_image.shape

dum = rescaled_image.reshape(1,-1)/255
dum.shape

with tf.Session() as sess:
    saver.restore(sess, "./model/my_model.ckpt")
    z = output_layer.eval(feed_dict = {X:dum, keep_prob:1.0})
    y_pred = np.argmax(Z, axis=1)
    print("Prediction for test image is {0}", format(y_pred))