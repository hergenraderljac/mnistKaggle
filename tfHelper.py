import tensorflow as tf
import numpy as np
import math
import pickle as p

def runSession(file, fileType=None, learning_rate = 0.0001, epochs = 10, batch_size=100, keep_prob=0.9, w2f = False):
	X_train, y_train, X_val, y_val = getDat(file, fileType)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		total_batch = int(X_train.shape[0]/batch_size)
		for epoch in range(epochs):
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
		print("Accuracy of validation set was", acc)
		if w2f:
			predict_n_format('test.pickle', sess)
		return acc

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
			print("Read in complete!")
			return X_train, y_train, X_val, y_val
		elif type == 'Test':
			print("Read in complete!")

			return



