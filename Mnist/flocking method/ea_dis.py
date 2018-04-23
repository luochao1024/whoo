import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tower
import time
from elastic_average_optimizer import \
  ElasticAverageOptimizer, ElasticAverageCustomGetter, GLOBAL_VARIABLE_NAME

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

def main():
    log_dir = '/logdir_%s_%s' % (FLAGS.job_name, FLAGS.task_index)
    # Server Setup
    cluster = tf.train.ClusterSpec({
	    'ps':['localhost:2222'],
	    'worker':['localhost:2223','localhost:2224']
	    }) #allows this node know about all other nodes
    if FLAGS.job_name == 'ps': #checks if parameter server
	with tf.device('/cpu:0'):
	    server = tf.train.Server(cluster,
	    job_name="ps",
	    task_index=FLAGS.task_index)
	    server.join()
    else:
	is_chief = (FLAGS.task_index == 0) #checks if this is the chief node
	server = tf.train.Server(cluster,job_name="worker",
	                        task_index=FLAGS.task_index)
	worker_device = "/job:worker/task:%d/gpu:0" % FLAGS.task_index
	ea_coustom = ElasticAverageCustomGetter(worker_device=worker_device)
	with variable_scope.variable_scope(
	    "", custom_getter=ea_coustom), ops.device(
	        device_setter.replica_device_setter(
	            worker_device=worker_device,
	            ps_device="/job:ps/task:0/cpu:0",
	            ps_tasks=1)):
	    loss = tower_loss()
	
	    sgd_opt = gradient_descent.GradientDescentOptimizer(1.0)
	    opt = ElasticAverageOptimizer(
	        opt=sgd_opt,
	        num_worker=2,
	        moving_rate=0.5,
	        communication_period=1,
	        ea_custom_getter=ea_coustom)	
	    grads_and_vars = opt.compute_gradients(loss)
	    train_op = opt.apply_gradients(grads_and_vars, global_step)
	    easgd_hook = opt.make_session_run_hook(is_chief, FLAGS.task_index)
	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.per_process_gpu_memory_fraction = 0.23
	stop_hook = [tf.train.StopAtStepHook(last_step = 1000)]
	with tf.train.MonitoredTrainingSession(
	    master=server.target,
	    checkpoint_dir=log_dir,
	    hooks=[easgd_hook, stop_hook],
	    config=config) as sess:
	    
	    while not sess.should_stop():
		loss, _ = sess.run(loss, train_op)
	    