from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tower
import time
import numpy as np
import argparse
from flocking_optimizer_dis import \
  FlockingOptimizer, FlockingCustomGetter, GLOBAL_VARIABLE_NAME,\
  LOCAL_VARIABLE_NAME, RECORD_LOCAL_VARIABLE_NAME

BATCH_SIZE = 32
NUM_TOWERS=4
ATTR = 1.0
REPU = 0.000
def main():
    port = 14976
    log_dir = './flocking_noncenter_%.3f_%.3f_%s_%s_r0.1' % (ATTR, REPU, FLAGS.job_name, FLAGS.task_index)
    config_ps = tf.ConfigProto(
            intra_op_parallelism_threads=3,
            inter_op_parallelism_threads=3)
    cluster = tf.train.ClusterSpec({
        'ps':['localhost:%d' % port],
        'worker':['localhost:%d' % (port+1), 'localhost:%d' % (port+2), 'localhost:%d' % (port+3), 'localhost:%d' % (port+4)]
        })
    if FLAGS.job_name == 'ps':
        with tf.device('/job:ps/task:0/cpu:0'):
            server = tf.train.Server(cluster, job_name='ps', task_index=FLAGS.task_index)
            server.join()
    
    else:
        is_chief = (FLAGS.task_index == 0)
        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)
        server = tf.train.Server(cluster, job_name='worker',
                                 task_index=FLAGS.task_index,
                                 config = config)
        with tf.device('/job:ps/task:0/cpu:0'):
            global_step = tf.Variable(0, name='global_step', trainable=False, )
            
        worker_device = '/job:worker/task:%d/cpu:0' % FLAGS.task_index
        f_getter = FlockingCustomGetter(num_towers=NUM_TOWERS, tower_index=FLAGS.task_index)
          
        x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
        y = tf.placeholder(tf.float32, [None, 10], name='y')
        logits = tower.tower(x, f_getter)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        
        
        sgd_opt = tf.train.GradientDescentOptimizer(0.01)
        opt = FlockingOptimizer(
            opt=sgd_opt,
            num_towers=NUM_TOWERS,
            tower_index = FLAGS.task_index,
            attraction=ATTR,
            repulsion=REPU)
        
        init = tf.global_variables_initializer()
        init_local = tf.variables_initializer(tf.local_variables())
        scaff = tf.train.Scaffold(init_op=init,local_init_op=[init_local])
        merged = tf.summary.merge_all()                              
        grads_and_vars = opt.compute_gradients(loss)
        train_op = opt.apply_gradients_and_flocking(grads_and_vars, global_step)
        stop_hook = tf.train.StopAtStepHook(last_step = 10)
        summary_hook = tf.train.SummarySaverHook(save_steps=20, output_dir=log_dir, summary_op=merged)
        with tf.train.MonitoredTrainingSession(master=server.target,
                                           scaffold = scaff,
                                           hooks=[stop_hook, summary_hook],
                                           config = config) as sess:
            for i in range(10000):
                if sess.should_stop():
                    break
                else:
                    batch_x, batch_y = mnist.train.next_batch(batch_size=BATCH_SIZE)
                    batch_n = np.reshape(batch_x, [-1, 28, 28, 1])
                    loss_value, accu_value, _ = sess.run([loss, accuracy, train_op], feed_dict={x:batch_n, y:batch_y})
                    print('tower_%d, loss is: %.4f, accuracy is %.3f' %(FLAGS.task_index, loss_value, accu_value))


                    # if i ==1999:
                    #     print("loss and accuracy")
                    #     test_x, test_y = mnist.test.next_batch(batch_size=1000)
                    #     test_n = np.reshape(test_x, [-1, 28, 28, 1])
                    #     lo, accu, summ = sess.run([loss, accuracy, merged], feed_dict={x:test_n, y:test_y})
                    #     print("loss is: ",lo)
                    #     print("accuracy is: ", accu)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
  # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    main()