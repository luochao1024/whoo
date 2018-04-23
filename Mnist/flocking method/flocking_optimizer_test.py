import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tower
import time
import flocking_optimizer as fopt
BATCH_SIZE = 10
NUM_TOWERS =4

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def tower_loss():
    batch_x, batch_y = mnist.train.next_batch(batch_size=BATCH_SIZE)
    batch_x_shaped = tf.reshape(batch_x, [-1, 28, 28, 1])
    ##None -> batch size can be any size, 784 -> flattened mnist image
    #x = tf.placeholder(np.float32, [10, 28, 28, 1])
    ##10 output classes
    #y = tf.placeholder(np.float32, shape=[10, 10], name='y_logits')
    logits = tower.tower(batch_x_shaped)
    # Reuse variables for the next tower.
    losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_y))   
    return losses

train_op_list=[None]*NUM_TOWERS
loss_list = [None]*NUM_TOWERS
with tf.device('/cpu:0'):
    global_step = tf.get_variable(initializer=0, trainable=False, name='global_step')
with tf.device('/gpu:0'):
    for tower_index, getter in enumerate(flocking_custom_getter_list):
        with tf.variable_scope('tower_%d' % tower_index, custom_getter=getter):
            loss = tower_loss()
            loss_list[tower_index] = loss
        
        sgd_opt = tf.train.GradientDescentOptimizer(1.0)
        opt = fopt.FlockingOptimizer(sgd_opt, NUM_TOWERS, tower_index, flocking_custom_getter_list)      
        grads_and_vars =  opt.compute_gradients(loss)
        train_op_list[tower_index] = opt.apply_gradients_and_flocking(tower_index=tower_index,
                                                                      grads_and_vars=grads_and_vars, 
                                                                      global_step=global_step)


aver_loss = tf.reduce_sum(loss_list)
condition = lambda m, n, step: tf.less(step, 100000000, name='step')

def train(i):
    with tf.control_dependencies([train_op_list[i]]):
        i += 0
    return i
    

body0 = lambda m, average_loss, step0: [train(0), loss_list[0], step0+1]
body1 = lambda m, average_loss, step1: [train(1), loss_list[1], step1+1]
body2 = lambda m, average_loss, step2: [train(3), loss_list[2], step2+1]
body3 = lambda m, average_loss, step3: [train(3), loss_list[3], step3+1]

train0, los0, stps0 = tf.while_loop(condition, body0, [train(0), loss_list[0], 0])
train1, los1, stps1 = tf.while_loop(condition, body1, [train(1), loss_list[1], 0])
train2, los2, stps2 = tf.while_loop(condition, body2, [train(2), loss_list[2], 0])
train3, los3, stps3 = tf.while_loop(condition, body3, [train(3), loss_list[3], 0])
train_group = tf.group(train0, train1, train2, train3)
loss_group = tf.group(los0, los1, los2, los3)

#def body0(step):
    #train_op0 = opt.apply_gradients_and_flocking(tower_index=tower_index,
                                                    #grads_and_vars=grads_and_vars_list[0], 
                                                    #global_step=global_step)
    #return global_step

#def body1(step):
    #train_op1 = opt.apply_gradients_and_flocking(tower_index=tower_index,
                                                    #grads_and_vars=grads_and_vars_list[1], 
                                                    #global_step=global_step)
    #return global_step

#def body2(step):
    #train_op2 = opt.apply_gradients_and_flocking(tower_index=tower_index,
                                                    #grads_and_vars=grads_and_vars_list[2], 
                                                    #global_step=global_step)
    #return global_step

#def body3(step):
    #train_op3 = opt.apply_gradients_and_flocking(tower_index=tower_index,
                                                    #grads_and_vars=grads_and_vars_list[3], 
                                                    #global_step=global_step)
    #return global_step
#loop_list=[None]*NUM_TOWERS
#body_list = [body0, body1, body2, body3]

#for tower_index in range(NUM_TOWERS):
    #loop_list[tower_index] = tf.while_loop(c,body_list[tower_index] , [global_step], name="loop_%d"%tower_index)

#aver_loss = tf.reduce_sum(loss_list)

local_init = tf.local_variables_initializer()
global_init = tf.global_variables_initializer()
config = tf.ConfigProto(allow_soft_placement = True)
with tf.Session(config=config) as sess:
    sess.run(local_init)
    sess.run(global_init)
    sess.run([train_group, loss_group])
    print(aver_loss.eval())
    
