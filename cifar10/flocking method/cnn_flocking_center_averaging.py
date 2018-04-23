import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import center_tower as tower
import time
import numpy as np
import argparse
from flocking_center_optimizer import \
  ElasticAverageOptimizer, ElasticAverageCustomGetter, GLOBAL_VARIABLE_NAME

BATCH_SIZE = 32

def main():
    port = 31281
    log_dir = './floking_center_logdir_%s_%s' % (FLAGS.job_name, FLAGS.task_index)
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
                                 task_index=FLAGS.task_index)
        worker_device = '/job:worker/task:%d/cpu:0' % FLAGS.task_index   
        ea_custom = ElasticAverageCustomGetter(worker_device=worker_device)
        with tf.variable_scope('', custom_getter=ea_custom), tf.device(
            tf.train.replica_device_setter(worker_device=worker_device,
                                           ps_device='/job:ps/task:0/cpu:0/',
                                           ps_tasks=1)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
            y = tf.placeholder(tf.float32, [None, 10], name='y')
            logits = tower.tower(x)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
            
        sgd_opt = tf.train.GradientDescentOptimizer(0.05)
        opt = ElasticAverageOptimizer(
            opt=sgd_opt,
            num_worker=4,
            moving_rate=0.001,
            communication_period=1,
            ea_custom_getter=ea_custom)
                      
                                  
        merged = tf.summary.merge_all()                              
        grads_and_vars = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(grads_and_vars, global_step)
        easgd_hook = opt.make_session_run_hook(is_chief, FLAGS.task_index)
        stop_hook = tf.train.StopAtStepHook(last_step = 400)
        summary_hook = tf.train.SummarySaverHook(save_steps=10, summary_op=merged)        
       
        with tf.train.MonitoredTrainingSession(master=server.target,
                                           hooks=[easgd_hook, stop_hook],
                                           config=config) as sess:
            count = 0
            
            
            while not sess.should_stop():
                batch_x, batch_y = mnist.train.next_batch(batch_size=BATCH_SIZE)
                batch_x = np.reshape(batch_x, [-1, 28, 28, 1])
                lo_train, _ = sess.run([loss, train_op],feed_dict={x:batch_x, y:batch_y})
                count += 1
                if not count%10:
                    print("lo_train is", lo_train)
                    test_x, test_y = mnist.test.next_batch(batch_size=1000)
                    test_x = np.reshape(test_x, [-1, 28, 28, 1])
                    lo, accu = sess.run([loss, accuracy], feed_dict={x:test_x, y:test_y})
                    print("number of examples used is:", count*BATCH_SIZE)
                    print("loss is: ",lo)
                    print("accuracy is: ", accu)
                    print()
        

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
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    main()
