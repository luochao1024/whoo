'''the mnist convolutional tower'''
import tensorflow as tf 
import numpy as np
import csv
# from flocking_optimizer_dis import GLOBAL_VARIABLE_NAME
# from tensorflow.python.ops import resource_variable_ops

w1 = np.array([0.0]*5*5*32)
w2 = np.array([0.0]*5*5*32*64)
w3 = np.array([0.0]*7*7*64*100)
w4 = np.array([0.0]*100*10)
ws=[w1, w2, w3, w4]
with open ("init_same_variables.txt", 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    l = [element for element in lines]
    
    for i in range(5*5*32):
        w1[i] = float(l[0][i])

    for i in range(5*5*32*64):
        w2[i] = float(l[1][i])

    for i in range(7*7*64*100):
        w3[i] = float(l[2][i])

    for i in range(100*10):
        w4[i] = float(l[3][i])

w1_shaped = np.reshape(w1, [5, 5, 1, 32])
w2_shaped = np.reshape(w2, [5, 5, 32, 64])
w3_shaped = np.reshape(w3, [7*7*64, 100])
w4_shaped = np.reshape(w4, [100, 10])

init1 = tf.constant_initializer(w1_shaped)
init2 = tf.constant_initializer(w2_shaped)
init3 = tf.constant_initializer(w3_shaped)
init4 = tf.constant_initializer(w4_shaped)

    
def tower(images):
    '''Build the mnist convolutional tower.
    
    Args:
      images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    
    Returns:
      Logits.
    '''
    w1 = np.array([0.0]*5*5*32)
    w2 = np.array([0.0]*5*5*32*64)
    w3 = np.array([0.0]*7*7*64*100)
    w4 = np.array([0.0]*100*10)
    ws=[w1, w2, w3, w4]
    with open ("init_same_variables.txt", 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        l = [element for element in lines]
        
        for i in range(5*5*32):
            w1[i] = float(l[0][i])

        for i in range(5*5*32*64):
            w2[i] = float(l[1][i])

        for i in range(7*7*64*100):
            w3[i] = float(l[2][i])

        for i in range(100*10):
            w4[i] = float(l[3][i])

    w1_shaped = np.reshape(w1, [5, 5, 1, 32])
    w2_shaped = np.reshape(w2, [5, 5, 32, 64])
    w3_shaped = np.reshape(w3, [7*7*64, 100])
    w4_shaped = np.reshape(w4, [100, 10])

    init1 = tf.constant_initializer(w1_shaped)
    init2 = tf.constant_initializer(w2_shaped)
    init3 = tf.constant_initializer(w3_shaped)
    init4 = tf.constant_initializer(w4_shaped)


    #conv1
   
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                              shape=[5, 5, 1, 32],
                              initializer = init1 )
        conv = tf.nn.conv2d(images, weights, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[32], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    #pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                              shape=[5, 5, 32, 64],
                              initializer=init2 )
        conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    #pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    #fully connected layer
    with tf.variable_scope('fully_conn') as scope:
        flattened = tf.reshape(pool2, [-1, 7*7*64])
        weights =tf.get_variable('weights',
                             shape=[7*7*64, 100],
                             initializer=init3 )
        biases = tf.get_variable('biases', shape=[100], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.matmul(flattened, weights) + biases
        fully_conn = tf.nn.relu(pre_activation, name=scope.name)
        #drop_out = tf.nn.dropout(fully_conn, keep_prob=0.5)

    #logits layer
    with tf.variable_scope('logits_layer') as scope:
        weights = tf.get_variable('weights',
                              shape=[100, 10],
                              initializer=init4 )
        biases = tf.get_variable('biases', shape=[10], initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(fully_conn, weights), biases)

    return logits

def global_variables_creator(num_towers):
    with tf.device(tf.train.replica_device_setter(ps_tasks=0, ps_device='/job:ps')):
        i = 0
        w1_tower_0 =  tf.Variable(name='w1_tower_%d' % i,
                                       initial_value=w1_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        w1_tower_1 =  tf.Variable(name='w1_tower_%d' % i,
                                       initial_value=w1_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        w1_tower_2 =  tf.Variable(name='w1_tower_%d' % i,
                                       initial_value=w1_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        w1_tower_3 =  tf.Variable(name='w1_tower_%d' % i,
                                       initial_value=w1_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])

        i = 0
        b1_tower_0 =  tf.Variable(name='b1_tower_%d' % i,
                                       initial_value=tf.zeros([32]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        b1_tower_1 =  tf.Variable(name='b1_tower_%d' % i,
                                       initial_value=tf.zeros([32]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        b1_tower_2 =  tf.Variable(name='b1_tower_%d' % i,
                                       initial_value=tf.zeros([32]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        b1_tower_3 =  tf.Variable(name='b1_tower_%d' % i,
                                       initial_value=tf.zeros([32]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])

        i = 0
        w2_tower_0 =  tf.Variable(name='w2_tower_%d' % i,
                                       initial_value=w2_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        w2_tower_1 =  tf.Variable(name='w2_tower_%d' % i,
                                       initial_value=w2_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        w2_tower_2 =  tf.Variable(name='w2_tower_%d' % i,
                                       initial_value=w2_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        w2_tower_3 =  tf.Variable(name='w2_tower_%d' % i,
                                       initial_value=w2_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])

        i =0
        b2_tower_0 =  tf.Variable(name='b2_tower_%d' % i,
                                       initial_value=tf.zeros([64]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        b2_tower_1 =  tf.Variable(name='b2_tower_%d' % i,
                                       initial_value=tf.zeros([64]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        b2_tower_2 =  tf.Variable(name='b2_tower_%d' % i,
                                       initial_value=tf.zeros([64]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        b2_tower_3 =  tf.Variable(name='b2_tower_%d' % i,
                                       initial_value=tf.zeros([64]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])

        i = 0
        w3_tower_0 =  tf.Variable(name='w3_tower_%d' % i,
                                       initial_value=w3_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        w3_tower_1 =  tf.Variable(name='w3_tower_%d' % i,
                                       initial_value=w3_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        w3_tower_2 =  tf.Variable(name='w3_tower_%d' % i,
                                       initial_value=w3_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        w3_tower_3 =  tf.Variable(name='w3_tower_%d' % i,
                                       initial_value=w3_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])

        i = 0
        b3_tower_0 =  tf.Variable(name='b3_tower_%d' % i,
                                       initial_value=tf.zeros([100]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        b3_tower_1 =  tf.Variable(name='b3_tower_%d' % i,
                                       initial_value=tf.zeros([100]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        b3_tower_2 =  tf.Variable(name='b3_tower_%d' % i,
                                       initial_value=tf.zeros([100]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        b3_tower_3 =  tf.Variable(name='b3_tower_%d' % i,
                                       initial_value=tf.zeros([100]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])


        i = 0
        w4_tower_0 =  tf.Variable(name='w4_tower_%d' % i,
                                       initial_value=w4_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        w4_tower_1 =  tf.Variable(name='w4_tower_%d' % i,
                                       initial_value=w4_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        w4_tower_2 =  tf.Variable(name='w4_tower_%d' % i,
                                       initial_value=w4_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        w4_tower_3 =  tf.Variable(name='w4_tower_%d' % i,
                                       initial_value=w4_shaped,
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])


        i = 0
        b4_tower_0 =  tf.Variable(name='b4_tower_%d' % i,
                                       initial_value=tf.zeros([10]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        b4_tower_1 =  tf.Variable(name='b4_tower_%d' % i,
                                       initial_value=tf.zeros([10]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        b4_tower_2 =  tf.Variable(name='b4_tower_%d' % i,
                                       initial_value=tf.zeros([10]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                    '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
        i += 1
        b4_tower_3 =  tf.Variable(name='b4_tower_%d' % i,
                                       initial_value=tf.zeros([10]),
                                       trainable=False,
                                       dtype=tf.float32,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                   '%s_tower_%d' % (GLOBAL_VARIABLE_NAME, i)])
