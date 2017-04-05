import argparse
import csv
import numpy
import sys
line = []
with open(opt['dataset']) as f:
    reader = csv.DictReader(f, delimiter=',')
b = open("op.csv", "r").readlines()
line = b[-2].split(',')
print line
