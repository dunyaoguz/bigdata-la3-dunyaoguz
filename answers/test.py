import os
import sys
import copy
import time
import random
import pyspark
from statistics import mean
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Window
from scipy.spatial import distance
import numpy as np
from all_states import all_states

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def distance2(filename, state1, state2):
    '''
    This function computes the squared Euclidean
    distance between two states.

    Return value: an integer.
    Test: tests/test_distance.py
    '''
    spark = init_spark()
    file = spark.read.text(filename).rdd
    plants = file.map(lambda x: x.value.split(',')[0]).distinct().collect()
    file = file.map(lambda x: (x.value.split(',')[0], x.value.split(',')[1:])) \
        .flatMap(lambda x: [(s, x[0]) for s in x[1]]) \
        .groupByKey() \
        .mapValues(list) \
        .map(lambda x: (x[0], {plant:1 for plant in x[1]}))
    states = file.filter(lambda x: x[0] in (state1, state2)) \
        .map(lambda x: (x[0], {plant:(1 if plant in x[1].keys() else 0) for plant in
                               plants})) \
        .map(lambda x: (x[0], [v for k, v in x[1].items()]))
    vectors = states.map(lambda x: x[1]).take(2)
    return np.sum([(vectors[0][i] - vectors[1][i])**2 for i in range(len(vectors[0]))])

def init_centroids(k, seed):
    '''
    This function randomly picks <k> states from the array in answers/all_states.py (you
    may import or copy this array to your code) using the random seed passed as
    argument and Python's 'random.sample' function.

    In the remainder, the centroids of the kmeans algorithm must be
    initialized using the method implemented here, perhaps using a line
    such as: `centroids = rdd.filter(lambda x: x[0] in
    init_states).collect()`, where 'rdd' is the RDD created in the data
    preparation task.

    Note that if your array of states has all the states, but not in the same
    order as the array in 'answers/all_states.py' you may fail the test case or
    have issues in the next questions.

    Return value: a list of <k> states.
    Test: tests/test_init_centroid.py
    '''
    random.seed(seed)
    centroids = random.sample(all_states, k)
    return centroids

def first_iter(filename, k, seed):
    '''
    This function assigns each state to its 'closest' class, where 'closest'
    means 'the class with the centroid closest to the tested state
    according to the distance defined in the distance function task'. Centroids
    must be initialized as in the previous task.

    Return value: a dictionary with <k> entries:
    - The key is a centroid.
    - The value is a list of states that are the closest to the centroid. The list should be alphabetically sorted.

    Test: tests/test_first_iter.py
    '''
    spark = init_spark()
    centroids = init_centroids(k, seed)
    file = spark.read.text(filename).rdd
    plants = file.map(lambda x: x.value.split(',')[0]).distinct().collect()
    file = file.map(lambda x: (x.value.split(',')[0], x.value.split(',')[1:]))\
               .flatMap(lambda x: [(s, x[0]) for s in x[1]])\
               .groupByKey()\
               .mapValues(list)\
               .map(lambda x: (x[0], {plant:1 for plant in x[1]}))\
               .map(lambda x: (x[0], [1 if plant in x[1].keys() else 0 for plant in plants]))
    # get the vectors of the centroids
    centroids_rdd = file.filter(lambda x: x[0] in centroids)\
                        .map(lambda x: {x[0]:x[1]})\
                        .collect()
    centroids_dict = dict(map(dict.popitem, centroids_rdd))
    # assign each state to a centroid
    assignments = file.map(lambda x: (x[0], x[1], {n:v for n, v in centroids_dict.items()}))\
                      .map(lambda x: (x[0], {n: np.sum([(x[1][i]-v[i])**2 for i in range(len(x[1]))])
                           for n, v in x[2].items()}))\
                      .map(lambda x: (x[0], x[1], sorted([v for k, v in x[1].items()])[0]))\
                      .map(lambda x: {x[0]: [k for k, v in x[1].items() if v == x[2]][0]})\
                      .collect()
    # fix the shape of the data
    assignments = dict(map(dict.popitem, assignments))
    final_dict = {}
    for centroid in centroids:
        closest = [state for state in all_states if assignments[state] == centroid]
        final_dict[centroid] = sorted(closest)
    return final_dict

spark = init_spark()
file = spark.read.text(os.path.join('.', 'data', 'plants.data')).rdd
plants = file.map(lambda x: x.value.split(',')[0]).distinct().collect()
file = file.map(lambda x: (x.value.split(',')[0], x.value.split(',')[1:]))\
            .flatMap(lambda x: [(s, x[0]) for s in x[1]])\
            .groupByKey()\
            .mapValues(list)\
            .map(lambda x: (x[0], {plant:1 for plant in x[1]}))\
            .map(lambda x: (x[0], [1 if plant in x[1].keys() else 0 for plant in plants]))

first_iteration = first_iter(os.path.join('.', 'data', 'plants.data'), 10, 123)
last_iteration = first_iteration.values()

while True:
    new_centroids = {}
    for i, cluster in enumerate(last_iteration):
        plant_vectors = file.filter(lambda x: x[0] in cluster)\
                            .map(lambda x: x[1])\
                            .collect()
        new_centroids[f'centroid_{i}'] = np.mean(plant_vectors, axis=0)

    new_assignments = file.map(lambda x: (x[0], x[1], {n:v for n, v in new_centroids.items()}))\
                          .map(lambda x: (x[0], {n: np.sum([(x[1][i]-v[i])**2 for i in range(len(x[1]))])
                                                     for n, v in x[2].items()}))\
                          .map(lambda x: (x[0], x[1], sorted([v for k, v in x[1].items()])[0]))\
                          .map(lambda x: {x[0]: [k for k, v in x[1].items() if v == x[2]][0]})\
                          .collect()

    new_assignments = dict(map(dict.popitem, new_assignments))

    final_list = []
    for centroid_name, centroid_vector in new_centroids.items():
        closest = [state for state in all_states if new_assignments[state] == centroid_name]
        final_list.append(sorted(closest))

    if final_list == last_iteration:
        break

    print(final_list)
    last_iteration = final_list
