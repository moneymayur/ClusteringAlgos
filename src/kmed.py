 #!/usr/bin/python
 
import argparse
import csv
import numpy
import sys
import sklearn
from sklearn import metrics
# SCRIPT PARAMETERS
parser = argparse.ArgumentParser(description='k-Medoids clustering implementation.')
parser.add_argument('--dataset', required=True, help='File in CSV format containing samples.')
parser.add_argument('--k', required=True, type=int, help='Number of clusters.')
parser.add_argument('--iterations', required=False, default=500, type=int, help='Maximum number of iterations.')
opt = vars(parser.parse_args())
 
# GLOBAL VARIABLES
targetClass = ''
attributeClass = {
 
}
data = []
medoids = []
k = opt['k']
N = 0 # N: size of training set
max_iter = int(opt['iterations'])
 
# LOAD DATASET
with open(opt['dataset']) as f:
    reader = csv.DictReader(f, delimiter='\t')
    # LOAD ATTRIBUTE TYPES (continuous, discrete, ignore)
    attributeClass = reader.next()
    # DETERMINE TARGET CLASS
    classLine = reader.next()
    for i in attributeClass:
        if classLine[i]=='class':
            targetClass = i
  
    # LOAD OBJECTS
    for row in reader:
        N += 1
        data.append(row)
  
       
# DISTANCE MEASURE & ERROR CRITERION
def manhattanDist(val1, val2): #val1 and val2 are str types
    a = val1.split(',')
    b = val2.split(',')
    a.pop()
    b.pop()
    a = numpy.asarray(a)
    b = numpy.asarray(b)
    a =a.astype(numpy.float)
    b =b.astype(numpy.float)
    return numpy.linalg.norm(abs(a-b))

def dist(val1, val2, attributeType):
	return manhattanDist(val1, val2)
 
 
def objectDist(obj1, obj2, attributeInfo):
    total=0
    n=0

    for att in attributeInfo:
        if (attributeInfo[att]!='' or attributeInfo[att] is not None) and obj1[att] is not None and obj2[att] is not None and obj1[att]!='' and obj2[att]!='' and obj1[att]!='NA' and obj2[att]!='NA':
        	total += float(dist(obj1[att], obj2[att], attributeInfo[att]))
        	n+=1
    if n==0: return None
    return total/n
 
def error_dist(medoids, data, attributeInfo):
    total=0
    for obj in data:
        # FIND NEAREST MEDOID AND COMPUTE DISTANCE
        minDist=numpy.inf
        for mi in medoids:
            m = data[mi]
            d=objectDist(obj,m, attributeInfo)
            if d<minDist: minDist=d
        
        total=total+minDist
    return total
 
# INITIALIZATION: SELECT RANDOM MEDOIDS
i=N/k
for j in range(k):
    medoids.append(j*i)

current_error = error_dist(medoids, data, attributeClass)
best_error = current_error
best_error_medoids = medoids[:]
delta=1
 
# MAIN LOOP
while max_iter>0 and delta>0:
    for mi in range(k):
       # print "current medoid:", mi
        for obj_i in range(N):
            if obj_i not in medoids:
                # SWAP
                # COPY medoids
                new_medoids = medoids[:]
                new_medoids[mi] = obj_i
                # COMPUTE NEW error
                new_error = error_dist(new_medoids, data, attributeClass)
                if new_error<best_error:
                    best_error = new_error
                    best_error_medoids = new_medoids[:]
    delta = current_error - best_error
    medoids = best_error_medoids[:]
    current_error = best_error
    
# DECIDE FINAL CLUSTERING
fields=attributeClass.keys()
fields.append(targetClass)
fields.append('cluster')
writer = csv.DictWriter(sys.stdout, fieldnames=fields, delimiter='\t')
writer.writeheader()
labels =list()
estimatedlabels =list()
for obj in data:
    # FIND NEAREST MEDOID AND ASSIGN CLUSTER
    minDist=numpy.inf
    nearestMedoid=''
    for mi in medoids:
        m = data[mi]
        d=objectDist(obj,m, attributeClass)
        if d<minDist:
            minDist=d
            nearestMedoid=mi
    nearestclass = str(data[nearestMedoid]).split(',').pop().replace("'}","")
    nearestclass =nearestclass.replace(" 'cluster': '","")
    labels.append(str(obj).split(',').pop().replace("'}",""))
    estimatedlabels.append(nearestclass)
   
labels= numpy.asarray(labels)
estimatedlabels= numpy.asarray(estimatedlabels)
print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels,estimatedlabels)
print "Completeness: %0.3f" % metrics.completeness_score(labels, estimatedlabels)
print "V-measure: %0.3f" % metrics.v_measure_score(labels,estimatedlabels)
print "Adjusted Rand Index: %0.3f" % \
    metrics.adjusted_rand_score(labels, estimatedlabels) 

