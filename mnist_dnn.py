import numpy as np
import csv
import tensorflow as tf
# from tensorflow.python import debug as tf_debug
# from tensorflow.examples.tutorials.mnist import input_data
import math
import pickle as p
import os
import tfHelper as th
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate = 0.0001
epochs = 10
batch_size = 100
z1 = 500
z2 = 250
z3 = 100
t_size = 0.8
v_size = 0.2
keep_prob = 0.9
writefile = 'ans.csv'

def csv2ndarray(filename):
	my_mat = np.genfromtxt(filename, delimiter=',', dtype='i8')
	if my_mat.shape[1] == 785:
		y = my_mat[1:,0]
		X = my_mat[1:,1:]
		Y = np.ndarray(shape=(y.shape[0],10))
		Y.fill(0)
		for i in range(y.shape[0]):
			Y[i,y[i]] = 1
		return X, Y
	else:
		return my_mat

def predict_n_format(filename, sess):
	if sess._closed:
		print("No tf session exists")
		return
	print("Session activated, reading test file...")
	X_test = p.load(open(filename,'rb'))
	print("Test file read in, predicting...")
	y_pred = sess.run(predictions, feed_dict={X:X_test})
	print("Writing predictions to " + writefile)
	with open(writefile, 'w', newline='') as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(['ImageID'] + ['Label'])
		for i in range(y_pred.shape[0]):
			writer.writerow([i+1] + [y_pred[i]])
	return

#Filename should not include extension
def getDat(filename, type=None):
	if type == None:
		print('Please provide whether the file is for training, or testing.')
		exit()
	try:
		file = filename + '.pickle'
		pickle_in = open(file, 'rb')
	except FileNotFoundError:
		print("File not found, please load file into pickle!")
		exit()
	else:
		if type == 'Train':
			my_mat = p.load(pickle_in)
			index = int(.8 * my_mat.shape[0])

			X_train = my_mat[:index,1:]
			X_val = my_mat[index:,1:]

			y_train	= np.zeros((index,10))
			y_val = np.zeros((my_mat.shape[0]-index,10))
			y_train[np.arange(index), my_mat[:index,0]] = 1
			y_val[np.arange(my_mat.shape[0]-index), my_mat[index:,0]] = 1
			return X_train, y_train, X_val, y_val
		elif type == 'Test':
			return

def hyperSearch():
	max = {'learning_rate': 0, 'keep_prob': 0, 'batch_size': 0, 'epochs': 0, 'acc': 0}
	init = tf.global_variables_initializer()
	for learning_rate in np.arange(0.0001, 0.0011, 0.0001):
		for keep_prob in np.arange(0.1, 1, 0.05):
			for batch_size in range(50,500,10):
				with tf.Session() as sess:
					sess.run(init)
					total_batch = int(X_train.shape[0]/batch_size)
					print(learning_rate, keep_prob, batch_size)
					for epoch in range(20):
						temp = 0
						for i in range(total_batch):
							# X_batch, y_batch = mnist.train.next_batch(batch_size=i)
							X_batch = X_train[i*batch_size:(i+1)*batch_size,:]
							y_batch = y_train[i*batch_size:(i+1)*batch_size,:]
							_, c = sess.run([optimizer,cross_entropy], feed_dict={X:X_batch, y:y_batch})
							if math.isnan(c)==False:
								temp = temp + c/total_batch
							else:
								print('Nope')
						print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(temp))
						acc = sess.run(accuracy, feed_dict={X: X_val, y: y_val})
						if acc > max['acc']:
							max['learning_rate'] = learning_rate
							max['keep_prob'] = keep_prob
							max['batch_size'] = batch_size
							max['epochs'] = epoch
							max['acc'] = acc
					print(max)
	return max

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784,z1], stddev = 0.03), name = "W1")
b1 = tf.Variable(tf.random_normal([z1]), name = "b1")

hidden1 = tf.add(tf.matmul(X,W1),b1)
H1 = tf.nn.dropout(tf.nn.relu(hidden1, name="H1"), keep_prob = keep_prob)

W2 = tf.Variable(tf.random_normal([z1,z2], stddev=0.03), name = "W2")
b2 = tf.Variable(tf.random_normal([z2]), name = "b2")

hidden2 = tf.add(tf.matmul(H1,W2), b2)
H2 = tf.nn.dropout(tf.nn.relu(hidden2, name="H2"),keep_prob=keep_prob)

W3 = tf.Variable(tf.random_normal([z2,z3], stddev=0.03), name = "W3")
b3 = tf.Variable(tf.random_normal([z3]), name = "b3")

hidden3 = tf.add(tf.matmul(H2,W3), b3)
H3 = tf.nn.dropout(tf.nn.relu(hidden3, name="H3"),keep_prob=keep_prob)

W_out = tf.Variable(tf.random_normal([z3,10], stddev = 0.03), name = "W4")
b_out = tf.Variable(tf.random_normal([10]), name = "b4")

hidden_out = tf.add(tf.matmul(H3,W_out),b_out)
y_ = tf.nn.softmax(hidden_out, name="H_out")
y_clipped = tf.clip_by_value(y_, 1e-10, 0.99999)

predictions = tf.argmax(y_,1)
 

cross_entropy = -tf.reduce_mean(tf.reduce_sum(y*tf.log(y_clipped) + (1 - y)*tf.log(1 - y_clipped), axis=1))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


# init = tf.global_variables_initializer()

# with tf.Session() as sess:
# 	sess.run(init)

# 	total_batch = int(X_train.shape[0]/batch_size)
# 	for epoch in range(epochs):
# 		temp = 0
# 		for i in range(total_batch):
# 			# X_batch, y_batch = mnist.train.next_batch(batch_size=i)
# 			X_batch = X_train[i*batch_size:(i+1)*batch_size,:]
# 			y_batch = y_train[i*batch_size:(i+1)*batch_size,:]
# 			_, c = sess.run([optimizer,cross_entropy], feed_dict={X:X_batch, y:y_batch})
# 			if math.isnan(c)==False:
# 				temp = temp + c/total_batch
# 			else:
# 				print('Nope')
# 		print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(temp))
# 	print(sess.run(accuracy, feed_dict={X: X_val, y: y_val}))
# 	# predict_n_format('test.pickle', sess)
# best_epoch = {}
# for epoch in range(20):
# 	#Better way to optimize epoch would be to check acc after each epoch, and just train, instead of starting
# 	#a new session and retraining all the time
# 	temp = 1./runSession(epochs=epoch) + math.exp(epoch/2000)
# 	print(temp)
# 	best_epoch.update({epoch: temp})
# print(min(best_epoch, key=best_epoch.get))
#TODO: Make function to perform hyperparamter search
print(th.runSession(file='train', fileType='Train'))





