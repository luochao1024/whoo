from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import init_tower as tower
import time
import numpy as np
import argparse

BATCH_SIZE = 4

def main():
    port = 24454
    log_dir = './cen_logdir_%s_%s' % (FLAGS.job_name, FLAGS.task_index)
    log_dir_test = './cen_test_logdir_%s_%s' % (FLAGS.job_name, FLAGS.task_index)
    config_ps = tf.ConfigProto(
            intra_op_parallelism_threads=3,
            inter_op_parallelism_threads=3)
    cluster = tf.train.ClusterSpec({
        'ps':['localhost:%d' % port],
        'worker':['localhost:%d' % (port+1), 'localhost:%d' % (port+2), 'localhost:%d' % (port+3), 'localhost:%d' % (port+4)]
        })
    if FLAGS.job_name == 'ps':
        with tf.device('/job:ps/task:0/cpu:0'):
            server = tf.train.Server(cluster, job_name='ps', task_index=FLAGS.task_index, config=config_ps)
            server.join()
    
    else:
        is_chief = (FLAGS.task_index == 0)
        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)
        server = tf.train.Server(cluster, job_name='worker',
                                 task_index=FLAGS.task_index,
                                 config=config)
                 
        worker_device = '/job:worker/task:%d/cpu:0' % FLAGS.task_index
    
        with tf.device(tf.train.replica_device_setter(worker_device=worker_device,
                                           ps_device='/job:ps/task:0/cpu:0/',
                                           ps_tasks=1)):
            global_step = tf.train.get_or_create_global_step()
            x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
            y = tf.placeholder(tf.float32, [None, 10], name='y')
            logits = tower.tower(x)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
        
        
        opt = tf.train.SyncReplicasOptimizer(
            tf.train.GradientDescentOptimizer(0.1),
            replicas_to_aggregate=4,
            total_num_replicas=4)
                      
        merged = tf.summary.merge_all()                              
        sync_replicas_hook = opt.make_session_run_hook(is_chief, num_tokens=0)
        train_op = opt.minimize(loss, global_step=global_step)
        stop_hook = tf.train.StopAtStepHook(last_step =200)
        
        summary_hook = tf.train.SummarySaverHook(save_steps=10, output_dir=log_dir, summary_op=merged)
        with tf.train.MonitoredTrainingSession(master=server.target,
                                           hooks=[sync_replicas_hook, stop_hook, summary_hook],
                                           config=config) as sess:
            f = open('./cen_logdir_%s_%s.txt' % (FLAGS.job_name, FLAGS.task_index), 'w')
            start_time = time.time()
            end_time = False
            for i in range(10000):
                print(FLAGS.task_index, 'i is', i)
                if sess.should_stop():
                    end_time = time.time()
                    print('worker', FLAGS.task_index, 'end all process')
                    print('time is', end_time - start_time)
                    break
                else:
                    batch_x, batch_y = mnist.train.next_batch(batch_size=BATCH_SIZE)
                    batch_n = np.reshape(batch_x, [-1, 28, 28, 1])
                    accu_value, loss_value, _ = sess.run([accuracy, loss, train_op], feed_dict={x:batch_n, y:batch_y})
                    #print('tower_%d, loss is: %.4f, accuracy is %.3f' %(FLAGS.task_index, loss_value, accu_value))
                    # print()
                    # print()
                    # print('i is', i)
                    # if i ==300: end_time = time.time() 
                    # if (not i % 5) and (FLAGS.task_index==2):
                    #     print("\n\nloss and accuracy")
                    #     test_x, test_y = mnist.test.next_batch(batch_size=5000)
                    #     test_n = np.reshape(test_x, [-1, 28, 28, 1])
                    #     lo, accu, summ = sess.run([loss, accuracy, merged], feed_dict={x:test_n, y:test_y})
                    #     print("\n\nloss of test dataset is: ",lo)
                    #     print("accuracy of test dataset is: %.3f\n\n"%accu)
                    #     if FLAGS.task_index==2:
                    #         f.write(str(lo)+' '+ str(accu)+ '\n')
                    #     if end_time:
                    #         f.write(str(end_time - start_time))
                    #         break

            

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
