from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cifar10
import time
from datetime import datetime
import numpy as np
import argparse

BATCH_SIZE = 4

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '',
                           """One of 'ps', 'worker' """)
tf.app.flags.DEFINE_integer('task_index', 0,
                           """Index of task within the job""")
tf.app.flags.DEFINE_string('train_dir', './multi_gpus_cifar10_syn',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

def train():
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
        gpu_options = tf.GPUOptions(allow_growth=True,
          allocator_type="BFC",
          visible_device_list="%d"%FLAGS.task_index)
        config = tf.ConfigProto(gpu_options=gpu_options,
          allow_soft_placement=True)
        server = tf.train.Server(cluster, job_name='worker',
                                 task_index=FLAGS.task_index,
                                 config=config)
                 
        worker_device = '/job:worker/task:%d/gpu:%d' % (FLAGS.task_index, FLAGS.task_index)
    
        with tf.device(tf.train.replica_device_setter(worker_device=worker_device,
                                           ps_device='/job:ps/task:0/cpu:0/',
                                           ps_tasks=1)):
            global_step = tf.train.get_or_create_global_step()

            # Get images and labels for CIFAR-10.
            # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
            # GPU and resulting in a slow down.
            with tf.device('/cpu:0'):
              images, labels = cifar10.distorted_inputs()

            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = cifar10.inference(images)

            # Calculate loss.
            loss = cifar10.loss(logits, labels)

            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            train_op, sync_replicas_hook = cifar10.train(loss, global_step, is_chief)
        
        
        
                                                      
            
            stop_hook = tf.train.StopAtStepHook(last_step =FLAGS.max_steps)

            class _LoggerHook(tf.train.SessionRunHook):
                """Logs loss and runtime."""

                def begin(self):
                  self._step = -1
                  self._start_time = time.time()

                def before_run(self, run_context):
                  self._step += 1
                  return tf.train.SessionRunArgs(loss)  # Asks for loss value.

                def after_run(self, run_context, run_values):
                  if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                         examples_per_sec, sec_per_batch))
        with tf.train.MonitoredTrainingSession(master=server.target,
                                           hooks=[sync_replicas_hook, stop_hook, tf.train.NanTensorHook(loss), _LoggerHook()],
                                           config=config) as sess:
            while not sess.should_stop():
                sess.run(train_op)
            
def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()