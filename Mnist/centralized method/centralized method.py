import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import mnist_network
import time

BATCH_SIZE = 10

def average_gradients(model_grads):
    '''calculate the average gradient for each shared variable across all models'''
    average_grads = []
    for grad_and_vars in zip(*model_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads=[]
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the model.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'model' dimension which we will average over below.
            grads.append(expanded_g)

    # Average over the 'model' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)    
    return average_grads

 
def train(mnist):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', shape=[], initializer=tf.constant_initializer(0), trainable = False)
        opt = tf.train.GradientDescentOptimizer(0.01)
        model_grads=[]
        with tf.device('/gpu:0'):
            for i in range(4):
                with tf.name_scope('cal_grad_%d' %(i)) as scope:
                    batch_x, batch_y = mnist.train.next_batch(batch_size=BATCH_SIZE)
                    batch_x_shaped = tf.reshape(batch_x, [-1, 28, 28, 1])
                    ##None -> batch size can be any size, 784 -> flattened mnist image
                    #x = tf.placeholder(np.float32, [10, 28, 28, 1])
                    ##10 output classes
                    #y = tf.placeholder(np.float32, shape=[10, 10], name='y_logits')
                    logits = mnist_network.model(batch_x_shaped)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()                    
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_y))
                    grad=opt.compute_gradients(loss)
                    model_grads.append(grad)
        
        average_grads = average_gradients(model_grads)
        apply_grad_op = opt.apply_gradients(average_grads, global_step=global_step)
        
        init = tf.global_variables_initializer()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            summary_writer = tf.summary.FileWriter('./mnist summary/', sess.graph)
            sess.run(init)
            for step in range(100):
                start_time = time.time()
                _, loss_value = sess.run([apply_grad_op, loss])
                duration = time.time() - start_time
                
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                
                if step %100 == 0:
                    num_examples_per_step = BATCH_SIZE*4
                    examples_per_sec = num_examples_per_step/duration
                    sec_per_batch = duration/4
                    print(loss_value)
                    print('elapsed time is {} for 100 steps'.format(duration))
                    print('%f: step %d, loss = %.2f (%.1f examples/sec) ' %(duration, step, loss_value, examples_per_sec))
                    



def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)
    
if __name__ == '__main__':
    main()
        