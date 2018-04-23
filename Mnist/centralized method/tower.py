'''the mnist convolutional tower'''
import tensorflow as tf 
        
def tower(images):
    '''Build the mnist convolutional tower.
    
    Args:
      images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    
    Returns:
      Logits.
    '''


    #conv1
    #regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                              shape=[5, 5, 1, 32],
                              initializer = tf.truncated_normal_initializer(
                                  stddev=tf.sqrt(2/(5.0*5*32)) ) )
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
                              initializer=tf.truncated_normal_initializer(
                                  stddev=tf.sqrt(2/(5.0*5*64)) ) )
        conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    x = tf.reshape(conv2, [-1, 14, 14, 64])
    #pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')
    
    #fully connected layer
    with tf.variable_scope('fully_conn') as scope:
        flattened = tf.reshape(pool2, [-1, 7*7*64])
        weights =tf.get_variable('weights',
                             shape=[7*7*64, 100],
                             initializer=tf.truncated_normal_initializer(
                                 stddev=1/tf.sqrt(7.0*7*64) ) )
        biases = tf.get_variable('biases', shape=[100], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.matmul(flattened, weights) + biases
        fully_conn = tf.nn.relu(pre_activation, name=scope.name)
        #drop_out = tf.nn.dropout(fully_conn, keep_prob=0.5)
        
    #logits layer
    with tf.variable_scope('logits_layer') as scope:
        weights = tf.get_variable('weights',
                              shape=[100, 10],
                              initializer=tf.truncated_normal_initializer(
                                  stddev=1/tf.sqrt(100.0) ) )
        biases = tf.get_variable('biases', shape=[10], initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(fully_conn, weights), biases)
    
    return logits
