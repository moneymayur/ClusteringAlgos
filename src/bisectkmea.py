import argparse
import csv
import numpy
import sys
import sklearn
from sklearn import metrics
from sklearn.preprocessing import scale
from time import time
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
parser = argparse.ArgumentParser(description='k-Medoids clustering implementation.')
parser.add_argument('--dataset', required=True, help='File in CSV format containing samples.')
parser.add_argument('--k', required=True, type=int, help='Number of clusters.')
parser.add_argument('--iterations', required=False, default=500, type=int, help='Maximum number of iterations.')
opt = vars(parser.parse_args())
N = 0 # N: size of training set
data = list()
medoids = []
K = opt['k']
labels = list()
N = 0 # N: size of training set
max_iter = int(opt['iterations'])
print type(K)
with open(opt['dataset']) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
    	labels.append(row.pop())
    	#print type(row)
    	data.append(row)

#print data
#print labels
r=[[]]   
km = KMeans(init='random', n_clusters= K, n_init=10).fit(data)
print km.labels_
for i in km.labels_
	if
