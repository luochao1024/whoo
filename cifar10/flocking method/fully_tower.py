'''the mnist convolutional tower'''
import tensorflow as tf 
        
def tower(images):
    '''Build the mnist convolutional tower.
    
    Args:
      images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    
    Returns:
      Logits.
    '''
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    with tf.variable_scope('fully_conn1') as scope:
        weights = tf.get_variable('weights',
                              shape=[784, 30],
                              regularizer=regularizer,
                              initializer=tf.truncated_normal_initializer(
                                  stddev=1/tf.sqrt(784.0) ) )
        tf.summary.histogram('weights', weights)
        biases = tf.get_variable('biases', 
                                 regularizer=regularizer,
                                 shape=[30], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.add(tf.matmul(images, weights), biases)
        fully_conn1 = tf.nn.sigmoid(pre_activation, name=scope.name)
        
    #with tf.variable_scope('fully_conn2') as scope:
        #weights = tf.get_variable('weights',
                              #shape=[30, 20],
                              #initializer=tf.truncated_normal_initializer(
                                  #stddev=1/tf.sqrt(30.0) ) )
        ##tf.summary.histogram('weights', weights)
        #biases = tf.get_variable('biases', shape=[20], initializer=tf.constant_initializer(0.0))
        #pre_activation = tf.add(tf.matmul(fully_conn1, weights), biases)
        #fully_conn2 = tf.nn.sigmoid(pre_activation, name=scope.name)  
        
    #logits layer
    with tf.variable_scope('logits_layer') as scope:
        weights = tf.get_variable('weights',
                              shape=[30, 10],
                              regularizer=regularizer,
                              initializer=tf.truncated_normal_initializer(
                                  stddev=1/tf.sqrt(30.0) ) )
        biases = tf.get_variable('biases', 
                                 regularizer=regularizer,
                                 shape=[10], initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(fully_conn1, weights), biases)
    
    return logits