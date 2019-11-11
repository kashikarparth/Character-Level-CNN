# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:39:59 2019

@author: parth
"""
import tensorflow as tf
import numpy as np
tf.reset_default_graph()



def network1(x, isTrain = True, reuse = False):
    
    with tf.variable_scope('CNN_global', reuse=reuse):
        
        shape1 = [70, 7, 1, 256]
        W1 = tf.Variable(tf.truncated_normal(shape=shape1, stddev=0.05), name="W1")
        b1 = tf.Variable(tf.constant(0.1, shape=[256]), name="b1")
        conv1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
        h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
        pooled1 = tf.nn.max_pool(h1,
            ksize=[1, 1, 3, 1],
            strides=[1, 1, 3, 1],
            padding='VALID',
            name="pool1")
        
        shape2 = [1, 7, 256, 256]
        W2 = tf.Variable(tf.truncated_normal(shape2, stddev=0.05), name="W2")
        b2 = tf.Variable(tf.constant(0.1, shape=[256]), name="b2")
        conv2 = tf.nn.conv2d(pooled1, W2, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
        h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
        pooled2 = tf.nn.max_pool(
            h2,
            ksize=[1, 1, 3, 1],
            strides=[1, 1, 3, 1],
            padding='VALID',
            name="pool2")
        
        shape3 = [1, 3, 256, 256]
        W3 = tf.Variable(tf.truncated_normal(shape3, stddev=0.05), name="W3")
        b3 = tf.Variable(tf.constant(0.1, shape=[256]), name="b3")
        conv3 = tf.nn.conv2d(pooled2, W3, strides=[1, 1, 1, 1], padding="VALID", name="conv3")
        h3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name="relu3")

        shape4 = [1, 3, 256, 256]
        W4 = tf.Variable(tf.truncated_normal(shape4, stddev=0.05), name="W4")
        b4 = tf.Variable(tf.constant(0.1, shape=[256]), name="b4")
        conv4 = tf.nn.conv2d(h3, W4, strides=[1, 1, 1, 1], padding="VALID", name="conv4")
        h4 = tf.nn.relu(tf.nn.bias_add(conv4, b4), name="relu4")
        
        shape5 = [1, 3, 256, 256]
        W5 = tf.Variable(tf.truncated_normal(shape5, stddev=0.05), name="W5")
        b5 = tf.Variable(tf.constant(0.1, shape=[256]), name="b5")
        conv5 = tf.nn.conv2d(h4, W5, strides=[1, 1, 1, 1], padding="VALID", name="conv5")
        h5 = tf.nn.relu(tf.nn.bias_add(conv5, b5), name="relu")
        
        shape6 = [1, 3, 256, 256]
        W6 = tf.Variable(tf.truncated_normal(shape6, stddev=0.05), name="W6")
        b6 = tf.Variable(tf.constant(0.1, shape=[256]), name="b6")
        conv6 = tf.nn.conv2d(h5, W6, strides=[1, 1, 1, 1], padding="VALID", name="conv6")
        h6 = tf.nn.relu(tf.nn.bias_add(conv6, b6), name="relu")
        pooled6 = tf.nn.max_pool(
            h6,
            ksize=[1, 1, 3, 1],
            strides=[1, 1, 3, 1],
            padding='VALID',
            name="pool6")
        
        num_features_total = 34 * 256
        h_pool_flat = tf.reshape(pooled6, [-1, num_features_total])
        
        drop1 = tf.nn.dropout(h_pool_flat,0.2)
        
        W_1 = tf.Variable(tf.truncated_normal([num_features_total, 1024], stddev=0.05), name="W_1")
        b_1 = tf.Variable(tf.constant(0.1, shape=[1024]), name="b_1")
        fc1 = tf.nn.relu(tf.nn.xw_plus_b(drop1, W_1, b_1), name="fc1")
        
        drop2 = tf.nn.dropout(fc1,0.2)
    
        W_2 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.05), name="W_2")
        b_2 = tf.Variable(tf.constant(0.1, shape=[1024]), name="b_2")
        fc2 = tf.nn.relu(tf.nn.xw_plus_b(drop2, W_2, b_2), name="fc2")
        
        W_3 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.05), name="W_3")
        b_3 = tf.Variable(tf.constant(0.1, shape=[10]), name="b_3")
        scores = tf.nn.xw_plus_b(fc2, W_3, b_3, name="output")
        predictions = tf.argmax(scores, 1, name="predictions")
        
    return scores,predictions

    
x = tf.convert_to_tensor(np.ones(shape=[1,70,1014,1]),dtype = tf.float32)

print(network1(x))
                 
                 
input_x = tf.placeholder(tf.float32, [None, 70, sequence_max_length, 1], name="input_x")
          