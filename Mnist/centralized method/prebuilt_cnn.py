import tensorflow as tf
import numpy as np
import tower
from tensorflow.examples.tutorials.mnist import input_data


def run_cnn():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Python optimisation variables
    learning_rate = 0.1
    epochs = 10
    batch_size = 50
    
    global_step = tf.train.get_or_create_global_step()
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')
    logits = tower.tower(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    
    # add an optimiser
    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # setup recording variables
    # add a summary to store the accuracy
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('C:\\Users\\Andy\\PycharmProjects')
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)    
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        for count in range(1000):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            batch_n = np.reshape(batch_x, [-1, 28, 28, 1])
            l, _ = sess.run([loss, train_op], feed_dict={x:batch_n, y:batch_y})
            if not count%10:
                test_x, test_y = mnist.test.next_batch(batch_size=1000)
                test_n = np.reshape(test_x, [-1, 28, 28, 1])
                lo, accu = sess.run([loss, accuracy], feed_dict={x:test_n, y:test_y})
                print('train loss is', l)
                print("loss is: ",lo)
                print("accuracy is: ", accu)
                print()        
        

if __name__ == "__main__":
    run_cnn()
