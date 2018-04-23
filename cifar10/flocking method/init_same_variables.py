import tensorflow as tf
import numpy as np
weights1 = tf.truncated_normal(shape=[5, 5, 1, 32],
                              stddev=tf.sqrt(2/(5.0*5*32)) )   
weights2 = tf.truncated_normal(shape=[5, 5, 32, 64],
                          stddev=tf.sqrt(2/(5.0*5*64)) ) 
weights3 =tf.truncated_normal(shape=[7*7*64, 100],
                         stddev=1/tf.sqrt(7.0*7*64) )   
weights4 = tf.truncated_normal(shape=[100, 10],
                          stddev=1/tf.sqrt(100.0) ) 

file_weights = open('init_same_variables.txt', 'w')
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	w1 = weights1.eval()
	w2 = weights2.eval()
	
	w3 = weights3.eval()
	w4 = weights4.eval()
	l = [w1, w2, w3, w4]
	for i in range(4):
		arr = np.reshape(l[i], [-1])
		print(len(arr))
		for ele in arr:
			file_weights.write(str(ele)+',')
		file_weights.write('\n')
