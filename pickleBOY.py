import csv
import pickle as p
import numpy as np

def pickleCSV(filename):
	file = filename + '.csv'
	my_mat = np.genfromtxt(file, delimiter=',', dtype='i8')
	my_mat = my_mat[1:,:]
	pickling_on = open(filename + '.pickle' ,"wb")
	p.dump(my_mat, pickling_on, protocol=3)
	pickling_on.close()





