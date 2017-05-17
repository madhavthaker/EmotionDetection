# Copyright 2017 Google, Inc. All Rights Reserved.
#
# ==============================================================================
import os
import tensorflow as tf
import sys
import urllib
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
from resizeimage import resizeimage
import glob

if sys.version_info[0] >= 3:
  from urllib.request import urlretrieve
else:
  from urllib import urlretrieve

LOGDIR = 'log3/'
GITHUB_URL ='https://raw.githubusercontent.com/mamcgrath/TensorBoard-TF-Dev-Summit-Tutorial/master/'

with tf.name_scope("Data_Initialization"):

    train = []
    for i in [1,2,3,4]:
        for filename in glob.glob('/Users/madhavthaker/Documents/CSCI63/Final Project/face-emoticon-master/data/ck+_scaled/*.png'): #assuming gif
            img=np.asarray(resizeimage.resize_cover(Image.open(filename),[128,128]))
            img_flat = img.reshape(img.size)
            train.append(img_flat)

    test = []
    for i in [1,2,3,4]:
        for filename in glob.glob('/Users/madhavthaker/Documents/CSCI63/Final Project/face-emoticon-master/data/ck+_test_scaled/*.png'): #assuming gif
            img=np.asarray(resizeimage.resize_cover(Image.open(filename),[128,128]))
            img_flat = img.reshape(img.size)
            test.append(img_flat)

    ### 0 = Angry, 2 = Disgust, 3 = Happy, 4 = Sad, 5 = Surprised,

    ### MNIST EMBEDDINGS ###
    ckp_labels = [5, 0, 3, 5, 4, 0, 1, 3, 5, 4, 0, 3, 5, 0, 1, 5, 4, 0, 0, 0, 2, 1, 3, 5, 0, 3, 5, 1, 3, 5, 0, 3, 5, 4, 0, 3, 5, 3, 1, 1, 0, 4, 5, 2, 1, 5, 3, 5, 1, 5, 3, 1, 5, 1, 5, 0, 1, 5, 3, 5, 1, 3, 0, 1, 5, 2, 3, 1, 5, 3, 1, 3, 1, 5, 3, 2, 5, 3, 1, 5, 3, 4, 0, 5, 0, 3, 1, 3, 2, 5, 1, 3, 5, 1, 5, 4, 0, 3, 1, 5, 1, 2, 5, 1, 3, 5, 3, 5, 1, 3, 5, 5, 3, 1, 1, 3, 4, 1, 5, 4, 1, 5, 0, 1, 3, 5, 2, 3, 5, 5, 3, 5, 1, 0, 1, 5, 3, 0, 5, 1, 0, 3, 5, 0, 3, 5, 3, 1, 4, 5, 1, 3, 5, 1, 3, 1, 3, 5, 1, 5, 0, 3, 5, 1, 1, 4, 1, 5, 1, 4, 1, 0, 1, 3, 5, 5, 0, 1, 0, 5, 4, 0, 5, 3, 5, 3, 5, 1, 3, 5, 2, 0, 5, 2, 0, 5, 2, 3, 4, 3, 2, 5, 1, 5, 0, 3, 0, 1, 3, 5, 0, 1, 3, 5, 0, 4, 3, 3, 1, 4, 2, 1, 3, 5, 5, 3, 0, 3, 1, 5, 5, 0, 3, 5, 3, 2, 5, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 0, 2, 4, 0, 7, 2, 0, 7, 0, 7, 2, 4, 4, 0, 2, 4, 7, 2, 5, 0, 3, 5, 4, 0, 1, 3, 5, 4, 0, 3, 5, 0, 1, 5, 4, 0, 0, 0, 2, 1, 3, 5, 0, 3, 5, 1, 3, 5, 0, 3, 5, 4, 0, 3, 5, 3, 1, 1, 0, 4, 5, 2, 1, 5, 3, 5, 1, 5, 3, 1, 5, 1, 5, 0, 1, 5, 3, 5, 1, 3, 0, 1, 5, 2, 3, 1, 5, 3, 1, 3, 1, 5, 3, 2, 5, 3, 1, 5, 3, 4, 0, 5, 0, 3, 1, 3, 2, 5, 1, 3, 5, 1, 5, 4, 0, 3, 1, 5, 1, 2, 5, 1, 3, 5, 3, 5, 1, 3, 5, 5, 3, 1, 1, 3, 4, 1, 5, 4, 1, 5, 0, 1, 3, 5, 2, 3, 5, 5, 3, 5, 1, 0, 1, 5, 3, 0, 5, 1, 0, 3, 5, 0, 3, 5, 3, 1, 4, 5, 1, 3, 5, 1, 3, 1, 3, 5, 1, 5, 0, 3, 5, 1, 1, 4, 1, 5, 1, 4, 1, 0, 1, 3, 5, 5, 0, 1, 0, 5, 4, 0, 5, 3, 5, 3, 5, 1, 3, 5, 2, 0, 5, 2, 0, 5, 2, 3, 4, 3, 2, 5, 1, 5, 0, 3, 0, 1, 3, 5, 0, 1, 3, 5, 0, 4, 3, 3, 1, 4, 2, 1, 3, 5, 5, 3, 0, 3, 1, 5, 5, 0, 3, 5, 3, 2, 5, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 0, 2, 4, 0, 7, 2, 0, 7, 0, 7, 2, 4, 4, 0, 2, 4, 7, 2, 5, 0, 3, 5, 4, 0, 1, 3, 5, 4, 0, 3, 5, 0, 1, 5, 4, 0, 0, 0, 2, 1, 3, 5, 0, 3, 5, 1, 3, 5, 0, 3, 5, 4, 0, 3, 5, 3, 1, 1, 0, 4, 5, 2, 1, 5, 3, 5, 1, 5, 3, 1, 5, 1, 5, 0, 1, 5, 3, 5, 1, 3, 0, 1, 5, 2, 3, 1, 5, 3, 1, 3, 1, 5, 3, 2, 5, 3, 1, 5, 3, 4, 0, 5, 0, 3, 1, 3, 2, 5, 1, 3, 5, 1, 5, 4, 0, 3, 1, 5, 1, 2, 5, 1, 3, 5, 3, 5, 1, 3, 5, 5, 3, 1, 1, 3, 4, 1, 5, 4, 1, 5, 0, 1, 3, 5, 2, 3, 5, 5, 3, 5, 1, 0, 1, 5, 3, 0, 5, 1, 0, 3, 5, 0, 3, 5, 3, 1, 4, 5, 1, 3, 5, 1, 3, 1, 3, 5, 1, 5, 0, 3, 5, 1, 1, 4, 1, 5, 1, 4, 1, 0, 1, 3, 5, 5, 0, 1, 0, 5, 4, 0, 5, 3, 5, 3, 5, 1, 3, 5, 2, 0, 5, 2, 0, 5, 2, 3, 4, 3, 2, 5, 1, 5, 0, 3, 0, 1, 3, 5, 0, 1, 3, 5, 0, 4, 3, 3, 1, 4, 2, 1, 3, 5, 5, 3, 0, 3, 1, 5, 5, 0, 3, 5, 3, 2, 5, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 0, 2, 4, 0, 7, 2, 0, 7, 0, 7, 2, 4, 4, 0, 2, 4, 7, 2, 5, 0, 3, 5, 4, 0, 1, 3, 5, 4, 0, 3, 5, 0, 1, 5, 4, 0, 0, 0, 2, 1, 3, 5, 0, 3, 5, 1, 3, 5, 0, 3, 5, 4, 0, 3, 5, 3, 1, 1, 0, 4, 5, 2, 1, 5, 3, 5, 1, 5, 3, 1, 5, 1, 5, 0, 1, 5, 3, 5, 1, 3, 0, 1, 5, 2, 3, 1, 5, 3, 1, 3, 1, 5, 3, 2, 5, 3, 1, 5, 3, 4, 0, 5, 0, 3, 1, 3, 2, 5, 1, 3, 5, 1, 5, 4, 0, 3, 1, 5, 1, 2, 5, 1, 3, 5, 3, 5, 1, 3, 5, 5, 3, 1, 1, 3, 4, 1, 5, 4, 1, 5, 0, 1, 3, 5, 2, 3, 5, 5, 3, 5, 1, 0, 1, 5, 3, 0, 5, 1, 0, 3, 5, 0, 3, 5, 3, 1, 4, 5, 1, 3, 5, 1, 3, 1, 3, 5, 1, 5, 0, 3, 5, 1, 1, 4, 1, 5, 1, 4, 1, 0, 1, 3, 5, 5, 0, 1, 0, 5, 4, 0, 5, 3, 5, 3, 5, 1, 3, 5, 2, 0, 5, 2, 0, 5, 2, 3, 4, 3, 2, 5, 1, 5, 0, 3, 0, 1, 3, 5, 0, 1, 3, 5, 0, 4, 3, 3, 1, 4, 2, 1, 3, 5, 5, 3, 0, 3, 1, 5, 5, 0, 3, 5, 3, 2, 5, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 0, 2, 4, 0, 7, 2, 0, 7, 0, 7, 2, 4, 4, 0, 2, 4, 7, 2]
    labels_train = np.array(ckp_labels).reshape(-1,1)

    ckp_test_labels = [1, 2, 3, 0, 5, 2, 0, 3, 2, 5, 1, 3, 2, 5, 5, 2, 5, 5, 3, 4, 5, 1, 3, 0, 3, 1, 0, 1, 5, 5, 1, 3, 5, 2, 4, 3, 5, 3, 3, 5, 2, 3, 3, 4, 5, 1, 5, 1, 5, 4, 0, 3, 4, 4, 7, 7, 7, 7, 4, 7, 7, 0, 1, 2, 3, 0, 5, 2, 0, 3, 2, 5, 1, 3, 2, 5, 5, 2, 5, 5, 3, 4, 5, 1, 3, 0, 3, 1, 0, 1, 5, 5, 1, 3, 5, 2, 4, 3, 5, 3, 3, 5, 2, 3, 3, 4, 5, 1, 5, 1, 5, 4, 0, 3, 4, 4, 7, 7, 7, 7, 4, 7, 7, 0, 1, 2, 3, 0, 5, 2, 0, 3, 2, 5, 1, 3, 2, 5, 5, 2, 5, 5, 3, 4, 5, 1, 3, 0, 3, 1, 0, 1, 5, 5, 1, 3, 5, 2, 4, 3, 5, 3, 3, 5, 2, 3, 3, 4, 5, 1, 5, 1, 5, 4, 0, 3, 4, 4, 7, 7, 7, 7, 4, 7, 7, 0, 1, 2, 3, 0, 5, 2, 0, 3, 2, 5, 1, 3, 2, 5, 5, 2, 5, 5, 3, 4, 5, 1, 3, 0, 3, 1, 0, 1, 5, 5, 1, 3, 5, 2, 4, 3, 5, 3, 3, 5, 2, 3, 3, 4, 5, 1, 5, 1, 5, 4, 0, 3, 4, 4, 7, 7, 7, 7, 4, 7, 7, 0];
    labels_test = np.array(ckp_test_labels).reshape(-1,1)
    print len(labels_test)

    enc = OneHotEncoder()
    enc.fit(labels_train)
    labels_final = enc.transform(labels_train).toarray()

    enc = OneHotEncoder()
    enc.fit(labels_test)
    labels_test_final = enc.transform(labels_test).toarray()

    train = np.asarray(train)
    test = np.asarray(test)
    print len(test)
    print len(labels_test_final)

# Add convolution layer
def conv_layer(input, size_in, size_out, name="conv"):
  with tf.name_scope(name):
    #w = tf.Variable(tf.zeros([5, 5, size_in, size_out]), name="W")
    #b = tf.Variable(tf.zeros([size_out]), name="B")
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Add fully connected layer
def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.nn.relu(tf.matmul(input, w) + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act


def mnist_model(learning_rate, use_two_conv, use_two_fc, hparam):

  tf.reset_default_graph()
  tf.set_random_seed(1)
  sess = tf.Session()

  # Setup placeholders, and reshape the data
  x = tf.placeholder(tf.float32, shape=[None, 128*128], name="x")
  x_image = tf.reshape(x, [-1, 128, 128, 1])
  tf.summary.image('input', x_image, 3)
  y = tf.placeholder(tf.float32, shape=[None, 7], name="labels")

  if use_two_conv:
    #x_image size:  (?, 128, 128, 1)
    #conv1 size:   (?, 64, 64, 96)
    #conv2size:  (?, 32, 32, 256)
    #conv3size:  (?, 16, 16, 384)
    #conv4size:  (?, 8, 8, 384)
    #convout_size:  (?, 4, 4, 384)

    print "x_image size: ", x_image.shape
    conv1 = conv_layer(x_image, 1, 86, "conv1")
    print "conv1 size:  ", conv1.shape
    conv2 = conv_layer(conv1, 86, 128, "conv2")
    print "conv2size: ", conv2.shape
    #conv3 = conv_layer(conv2, 128, 256, "conv3")
    #print "conv3size: ", conv3.shape
    #conv4 = conv_layer(conv3, 256, 256, "conv4")
    #print "conv4size: ", conv4.shape
    #conv_out = conv_layer(conv4, 256, 256, "conv5")
    #print "convout_size: ", conv_out.shape

  else:
    conv1 = conv_layer(x_image, 1, 64, "conv")
    conv_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME") #adding padding "VALID" means no padding
  flattened = tf.reshape(conv2, [-1, 32 * 32 * 128])


  if use_two_fc:
    fc1 = fc_layer(flattened, 32 * 32 * 128, 40, "fc1")
    logits = fc_layer(fc1, 40, 7, "fc2")
  else:
    embedding_input = flattened
    embedding_size = 7*7*64
    logits = fc_layer(flattened, 7*7*64, 10, "fc")

  with tf.name_scope("xent"):
    xent = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="xent")
    tf.summary.scalar("xent", xent)

  with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

  with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(y, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

  summ = tf.summary.merge_all()

  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter('/Users/madhavthaker/Documents/CSCI63/Final Project/temp')
  writer.add_graph(sess.graph)

  for i in range(300):
    batch_index_train = random.sample(range(0,len(train)),300)
    batch_index_test = random.sample(range(0,len(test)),100)

    if i % 5 == 0:
      [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: train[batch_index_train], y: labels_final[batch_index_train]})
      [test_accuracy, s1] = sess.run([accuracy, summ], feed_dict={x: test[batch_index_test], y: labels_test_final[batch_index_test]})
      writer.add_summary(s, i)
      writer.add_summary(s1, i)
      print ("train accuracy (test accuracy): {0} ({1})".format(train_accuracy, test_accuracy))
    sess.run(train_step, feed_dict={x: train[batch_index_train], y: labels_final[batch_index_train]})
  writer.close()
  sess.close()

  #[test_accuracy_final, s] = sess.run([accuracy, summ], feed_dict={x: test, y: labels_test_final})
  #print ("Overall test accuracy: {0}".format(test_accuracy_final))


def main():
    # Actually run with the new settings
    mnist_model(0.03, True, True, None)



if __name__ == '__main__':
  main()
