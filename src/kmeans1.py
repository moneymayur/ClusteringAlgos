import argparse
import csv
import numpy
import sys
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
parser = argparse.ArgumentParser(description='k-Medoids clustering implementation.')
parser.add_argument('--dataset', required=True, help='File in CSV format containing samples.')
parser.add_argument('--k', required=True, type=int, help='Number of clusters.')
parser.add_argument('--iterations', required=False, default=500, type=int, help='Maximum number of iterations.')
opt = vars(parser.parse_args())
#declarations
data = list()
medoids = []
K = opt['k']
labels = list()
N = 0 # N: size of training set
max_iter = int(opt['iterations'])
with open(opt['dataset']) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
    	labels.append(row.pop())  # storing labels and data separately
    	data.append(row)
# Executing K means with random and its measures using scikit learn
km = KMeans(init='random', n_clusters= K, n_init=10).fit(data)
print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_)
print "Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_)
print "V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_)
print "Adjusted Rand Index: %0.3f" % \
    metrics.adjusted_rand_score(labels, km.labels_)
# Executing K means ++ and its measures using scikit learn
km = KMeans(init='k-means++', n_clusters= K, n_init=10).fit(data)
print 'printing for k++'
print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_)
print "Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_)
print "V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_ )
print "Adjusted Rand Index: %0.3f" % \
    metrics.adjusted_rand_score(labels, km.labels_) 
