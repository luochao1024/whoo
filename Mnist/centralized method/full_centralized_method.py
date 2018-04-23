import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import fully_tower
import time
import numpy as np
import argparse

BATCH_SIZE = 32

# def tower_loss_accuracy():
#     batch_x, batch_y = mnist.train.next_batch(batch_size=BATCH_SIZE)
#     batch_x_shaped = tf.reshape(batch_x, [-1, 28, 28, 1])
#     ##None -> batch size can be any size, 784 -> flattened mnist image
#     #x = tf.placeholder(np.float32, [10, 28, 28, 1])
#     ##10 output classes
#     #y = tf.placeholder(np.float32, shape=[10, 10], name='y_logits')
#     logits = tower.tower(batch_x_shaped)
#     # Reuse variables for the next tower.
#     losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_y))
#     correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(batch_y, 1))
#     accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     return losses, accuracy_

def main():
    port = 38461
    log_dir = './logdir_%s_%s' % (FLAGS.job_name, FLAGS.task_index)
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
        server = tf.train.Server(cluster, job_name='worker',
                                 task_index=FLAGS.task_index)
        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)         
        worker_device = '/job:worker/task:%d/cpu:0' % FLAGS.task_index
    
        with tf.device(tf.train.replica_device_setter(worker_device=worker_device,
                                           ps_device='/job:ps/task:0/cpu:0/',
                                           ps_tasks=1)):
            global_step = tf.train.get_or_create_global_step()
            x = tf.placeholder(tf.float32, [None, 784], name='x')
            y = tf.placeholder(tf.float32, [None, 10], name='y')
            logits = fully_tower.tower(x)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
        
        
        opt = tf.train.SyncReplicasOptimizer(
            tf.train.GradientDescentOptimizer(0.01),
            replicas_to_aggregate=4,
            total_num_replicas=4)
                      
        merged = tf.summary.merge_all()                              
        sync_replicas_hook = opt.make_session_run_hook(is_chief)
        train_op = opt.minimize(loss, global_step=global_step)
        stop_hook = tf.train.StopAtStepHook(last_step = 200)
        summary_hook = tf.train.SummarySaverHook(save_steps=10, output_dir=log_dir, summary_op=merged)
        with tf.train.MonitoredTrainingSession(master=server.target,
                                           hooks=[sync_replicas_hook, stop_hook],
                                           config=config) as sess:
            count=0
            while not sess.should_stop():
                count+=1
                batch_x, batch_y = mnist.train.next_batch(batch_size=BATCH_SIZE)
                sess.run(train_op, feed_dict={x:batch_x, y:batch_y})
                if not count%2:
                    test_x, test_y = mnist.test.next_batch(batch_size=1000)
                    lo, accu = sess.run([loss, accuracy], feed_dict={x:test_x, y:test_y})
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
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    main()
