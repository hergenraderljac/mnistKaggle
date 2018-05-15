import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.5
epochs = 10
batch_size = 100
#This tells tensorflow when building the network to expect x to be a 
#variable with data type float32, and is a tensor with an undefined number 
#of rows and 784 collumns
x = tf.placeholder(tf.float32, [None, 784])
#This tells tensorflow when building the network to expect y to be a 
#variable with data type float32, and is a tensor with an undefined number 
#of rows and 10 collumns
y = tf.placeholder(tf.float32, [None, 10])

#This declares the variable W1 to be a tensor of size 784,300 
W1 = tf.Variable(tf.random_normal([784,300], stddev=0.03), name="W1")
#b1 is a bias layer!
b1 = tf.Variable(tf.random_normal([300]), name="b1")

W2 = tf.Variable(tf.random_normal([300,10], stddev=0.03), name="W2")
b2 = tf.Variable(tf.random_normal([10]), name="b2")

#Manually adds x*W1' to the varibale b1
hidden_out = tf.add(tf.matmul(x,W1), b1)
hidden_out = tf.nn.relu(hidden_out)

y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out,W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.99999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y*tf.log(y_clipped) + (1 - y)*tf.log(1 - y_clipped), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

init_op = tf.global_variables_initializer()

correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


with tf.Session() as sess:
	#Uncomment to go into debug mode
	# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	sess.run(init_op)
	total_batch = int(len(mnist.train.labels)/batch_size)
	for epoch in range(epochs):
		avg_cost = 0
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
			_, c = sess.run([optimizer, cross_entropy], feed_dict={x:batch_x, y:batch_y})
			avg_cost += c/total_batch
		print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

