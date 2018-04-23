'''the mnist convolutional tower'''
import tensorflow as tf 
        
def tower(images):
    '''Build the mnist convolutional tower.
    
    Args:
      images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    
    Returns:
      Logits.
    '''
    with tf.variable_scope('fully_conn1') as scope:
        weights = tf.get_variable('weights',
                              shape=[784, 30],
                              initializer=tf.truncated_normal_initializer(
                                  stddev=1/tf.sqrt(784.0) ) )
        biases = tf.get_variable('biases', shape=[30], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.add(tf.matmul(images, weights), biases)
        fully_conn = tf.nn.sigmoid(pre_activation, name=scope.name)
        
    #logits layer
    with tf.variable_scope('logits_layer') as scope:
        weights = tf.get_variable('weights',
                              shape=[30, 10],
                              initializer=tf.truncated_normal_initializer(
                                  stddev=1/tf.sqrt(30.0) ) )
        biases = tf.get_variable('biases', shape=[10], initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(fully_conn, weights), biases)
    
    return logits
