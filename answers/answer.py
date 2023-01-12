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
import numpy as np
from all_states import all_states

'''
INTRODUCTION

With this assignment you will get a practical hands-on of frequent 
itemsets and clustering algorithms in Spark. Before starting, you may 
want to review the following definitions and algorithms:
* Frequent itemsets: Market-basket model, association rules, confidence, interest.
* Clustering: kmeans clustering algorithm and its Spark implementation.

DATASET

We will use the dataset at 
https://archive.ics.uci.edu/ml/datasets/Plants, extracted from the USDA 
plant dataset. This dataset lists the plants found in US and Canadian 
states.

The dataset is available in data/plants.data, in CSV format. Every line 
in this file contains a tuple where the first element is the name of a 
plant, and the remaining elements are the states in which the plant is 
found. State abbreviations are in data/stateabbr.txt for your 
information.
'''

'''
HELPER FUNCTIONS

These functions are here to help you. Instructions will tell you when
you should use them. Don't modify them!
'''

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def toCSVLineRDD(rdd):
    a = rdd.map(lambda row: ",".join([str(elt) for elt in row])) \
        .reduce(lambda x, y: '\n'.join([x, y]))
    return a + '\n'

def toCSVLine(data):
    if isinstance(data, RDD):
        if data.count() > 0:
            return toCSVLineRDD(data)
        else:
            return ""
    elif isinstance(data, DataFrame):
        if data.count() > 0:
            return toCSVLineRDD(data.rdd)
        else:
            return ""
    return None


'''
PART 1: FREQUENT ITEMSETS

Here we will seek to identify association rules between states to 
associate them based on the plants that they contain. For instance, 
"[A, B] => C" will mean that "plants found in states A and B are likely 
to be found in state C". We adopt a market-basket model where the 
baskets are the plants and the items are the states. This example 
intentionally uses the market-basket model outside of its traditional 
scope to show how frequent itemset mining can be used in a variety of 
contexts.
'''

def data_frame(filename, n):
    '''
    Write a function that returns a CSV string representing the first 
    <n> rows of a DataFrame with the following columns,
    ordered by increasing values of <id>:
    1. <id>: the id of the basket in the data file, i.e., its line number - 1 (ids start at 0).
    2. <plant>: the name of the plant associated to basket.
    3. <items>: the items (states) in the basket, ordered as in the data file.

    Return value: a CSV string. Using function toCSVLine on the right 
                  DataFrame should return the correct answer.
    Test file: tests/test_data_frame.py
    '''
    spark = init_spark()
    file = spark.read.text(filename)
    # process data
    file = file.withColumn("row", split(file.value, ","))
    file = file.withColumn("plant", file.row[0])
    file = file.withColumn("no_states", size(file.row)-1)
    file = file.withColumn("states", expr("slice(row, 2, no_states)"))
    # transform into a new df
    df = file.groupby("plant").agg(collect_set("states").alias("states")).orderBy('plant')
    df = df.withColumn("states", flatten("states"))
    window = Window.orderBy(df.plant)
    df = df.withColumn("id", row_number().over(window)-1)
    df = df.select("id", "plant", "states").limit(n)
    return toCSVLine(df)

def frequent_itemsets(filename, n, s, c):
    '''
    Using the FP-Growth algorithm from the ML library (see
    http://spark.apache.org/docs/latest/ml-frequent-pattern-mining.html), 
    write a function that returns the first <n> frequent itemsets 
    obtained using min support <s> and min confidence <c> (parameters 
    of the FP-Growth model), sorted by (1) descending itemset size, and 
    (2) descending frequency. The FP-Growth model should be applied to 
    the DataFrame computed in the previous task. 
    
    Return value: a CSV string. As before, using toCSVLine may help.
    Test: tests/test_frequent_items.py
    '''
    # output from the first task
    spark = init_spark()
    file = spark.read.text(filename)
    # process data
    file = file.withColumn("row", split(file.value, ","))
    file = file.withColumn("plant", file.row[0])
    file = file.withColumn("no_states", size(file.row)-1)
    file = file.withColumn("states", expr("slice(row, 2, no_states)"))
    # transform into a new df
    df = file.groupby("plant").agg(collect_set("states").alias("states")).orderBy('plant')
    df = df.withColumn("states", flatten("states"))
    window = Window.orderBy(df.plant)
    df = df.withColumn("id", row_number().over(window)-1)
    # apply FPGrowth algorithm
    fpGrowth = FPGrowth(itemsCol="states", minSupport=s, minConfidence=c)
    model = fpGrowth.fit(df)
    frequent_itemsets = model.freqItemsets.withColumn("no_items", size(model.freqItemsets.items))
    frequent_itemsets = frequent_itemsets.orderBy(col("no_items").desc(), col("freq").desc())
    return toCSVLine(frequent_itemsets.drop("no_items").limit(n))

def association_rules(filename, n, s, c):
    '''
    Using the same FP-Growth algorithm, write a script that returns the 
    first <n> association rules obtained using min support <s> and min 
    confidence <c> (parameters of the FP-Growth model), sorted by (1) 
    descending antecedent size in association rule, and (2) descending 
    confidence.

    Return value: a CSV string.
    Test: tests/test_association_rules.py
    '''
    # output from the first task
    spark = init_spark()
    file = spark.read.text(filename)
    # process data
    file = file.withColumn("row", split(file.value, ","))
    file = file.withColumn("plant", file.row[0])
    file = file.withColumn("no_states", size(file.row)-1)
    file = file.withColumn("states", expr("slice(row, 2, no_states)"))
    # transform into a new df
    df = file.groupby("plant").agg(collect_set("states").alias("states")).orderBy('plant')
    df = df.withColumn("states", flatten("states"))
    window = Window.orderBy(df.plant)
    df = df.withColumn("id", row_number().over(window)-1)
    # apply FPGrowth algorithm
    fpGrowth = FPGrowth(itemsCol="states", minSupport=s, minConfidence=c)
    model = fpGrowth.fit(df)
    assc_rules = model.associationRules.withColumn("no_antecedent",
                                                   size(model.associationRules.antecedent))
    assc_rules = assc_rules.orderBy(col("no_antecedent").desc(), col("confidence").desc())
    return toCSVLine(assc_rules.select("antecedent", "consequent", "confidence").limit(n))

def interests(filename, n, s, c):
    '''
    Using the same FP-Growth algorithm, write a script that computes 
    the interest of association rules (interest = |confidence - 
    frequency(consequent)|; note the absolute value)  obtained using 
    min support <s> and min confidence <c> (parameters of the FP-Growth 
    model), and prints the first <n> rules sorted by (1) descending 
    antecedent size in association rule, and (2) descending interest.

    Return value: a CSV string.
    Test: tests/test_interests.py
    '''
    # output from the first task
    spark = init_spark()
    file = spark.read.text(filename)
    # process data
    file = file.withColumn("row", split(file.value, ","))
    file = file.withColumn("plant", file.row[0])
    file = file.withColumn("no_states", size(file.row)-1)
    file = file.withColumn("states", expr("slice(row, 2, no_states)"))
    # transform into a new df
    df = file.groupby("plant").agg(collect_set("states").alias("states")).orderBy('plant')
    df = df.withColumn("states", flatten("states"))
    window = Window.orderBy(df.plant)
    df = df.withColumn("id", row_number().over(window)-1)
    # apply FPGrowth algorithm
    fpGrowth = FPGrowth(itemsCol="states", minSupport=s, minConfidence=c)
    model = fpGrowth.fit(df)
    frequent_itemsets = model.freqItemsets.withColumn("no_items", size(model.freqItemsets.items))
    assc_rules = model.associationRules.withColumn("no_antecedent",
                                                   size(model.associationRules.antecedent))
    both = assc_rules.join(frequent_itemsets, assc_rules.consequent == frequent_itemsets.items)
    basket_size = df.count()
    both = both.withColumn('no_basket', lit(basket_size))
    both = both.withColumn('consequent_frequency', both.freq / both.no_basket)
    both = both.withColumn('interest', both.confidence - both.consequent_frequency)
    both = both.orderBy(col("no_antecedent").desc(), col("interest").desc()).limit(n)
    return toCSVLine(both.select('antecedent', 'consequent', 'confidence', 'items', 'freq',
                                 'interest'))

'''
PART 2: CLUSTERING

We will now cluster the states based on the plants that they contain.
We will reimplement and use the kmeans algorithm. States will be 
represented by a vector of binary components (0/1) of dimension D, 
where D is the number of plants in the data file. Coordinate i in a 
state vector will be 1 if and only if the ith plant in the dataset was 
found in the state (plants are ordered alphabetically, as in the 
dataset). For simplicity, we will initialize the kmeans algorithm 
randomly.

An example of clustering result can be visualized in states.png in this 
repository. This image was obtained with R's 'maps' package (Canadian 
provinces, Alaska and Hawaii couldn't be represented and a different 
seed than used in the tests was used). The classes seem to make sense 
from a geographical point of view!
'''

def data_preparation(filename, plant, state):
    '''
    This function creates an RDD in which every element is a tuple with 
    the state as first element and a dictionary representing a vector 
    of plant as a second element:
    (name of the state, {dictionary})

    The dictionary should contains the plant names as keys. The 
    corresponding values should be 1 if the plant occurs in the state 
    represented by the tuple.

    You are strongly encouraged to use the RDD created here in the 
    remainder of the assignment.

    Return value: True if the plant occurs in the state and False otherwise.
    Test: tests/test_data_preparation.py
    '''
    spark = init_spark()
    file = spark.read.text(filename).rdd
    file = file.map(lambda x: (x.value.split(',')[0], x.value.split(',')[1:]))\
               .flatMap(lambda x: [(s, x[0]) for s in x[1]])\
               .groupByKey()\
               .mapValues(list)\
               .map(lambda x: (x[0], {plant:1 for plant in x[1]}))
    test = file.filter(lambda x: x[0] == state)\
               .map(lambda x: plant in x[1].keys())\
               .collect()
    return test[0]

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
    file = file.map(lambda x: (x.value.split(',')[0], x.value.split(',')[1:]))\
               .flatMap(lambda x: [(s, x[0]) for s in x[1]])\
               .groupByKey()\
               .mapValues(list)\
               .map(lambda x: (x[0], {plant:1 for plant in x[1]}))
    states = file.filter(lambda x: x[0] in (state1, state2))\
                 .map(lambda x: (x[0], {plant:(1 if plant in x[1].keys() else 0) for plant in
                                        plants}))\
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

def kmeans(filename, k, seed):
    '''
    This function:
    1. Initializes <k> centroids.
    2. Assigns states to these centroids as in the previous task.
    3. Updates the centroids based on the assignments in 2.
    4. Goes to step 2 if the assignments have not changed since the previous iteration.
    5. Returns the <k> classes.

    Note: You should use the list of states provided in all_states.py to ensure that the same initialization is made.
    
    Return value: a list of lists where each sub-list contains all states (alphabetically sorted) of one class.
                  Example: [["qc", "on"], ["az", "ca"]] has two 
                  classes: the first one contains the states "qc" and 
                  "on", and the second one contains the states "az" 
                  and "ca".
    Test file: tests/test_kmeans.py
    '''
    # prepare the file
    spark = init_spark()
    file = spark.read.text(filename).rdd
    plants = file.map(lambda x: x.value.split(',')[0]).distinct().collect()
    file = file.map(lambda x: (x.value.split(',')[0], x.value.split(',')[1:]))\
               .flatMap(lambda x: [(s, x[0]) for s in x[1]])\
               .groupByKey()\
               .mapValues(list)\
               .map(lambda x: (x[0], {plant:1 for plant in x[1]}))\
               .map(lambda x: (x[0], [1 if plant in x[1].keys() else 0 for plant in plants]))
    # get first iteration
    first_iteration = first_iter(filename, k, seed)
    last_iteration = first_iteration.values()
    while True:
        # compute the new centroids
        new_centroids = {}
        for i, cluster in enumerate(last_iteration):
            plant_vectors = file.filter(lambda x: x[0] in cluster) \
                .map(lambda x: x[1]) \
                .collect()
            new_centroids[f'centroid_{i}'] = np.mean(plant_vectors, axis=0)
        # find the new assignments
        new_assignments = file.map(lambda x: (x[0], x[1], {n:v for n, v in new_centroids.items()})) \
            .map(lambda x: (x[0], {n: np.sum([(x[1][i]-v[i])**2 for i in range(len(x[1]))])
                                   for n, v in x[2].items()})) \
            .map(lambda x: (x[0], x[1], sorted([v for k, v in x[1].items()])[0])) \
            .map(lambda x: {x[0]: [k for k, v in x[1].items() if v == x[2]][0]}) \
            .collect()
        new_assignments = dict(map(dict.popitem, new_assignments))
        # turn assignments into list of lists
        final_list = []
        for centroid_name, centroid_vector in new_centroids.items():
            closest = [state for state in all_states if new_assignments[state] == centroid_name]
            final_list.append(sorted(closest))
        # if last iteration is same as the current iteration, the assignments converged
        # break from the loops
        if final_list == last_iteration:
            break
        # otherwise, continue computing new centroids
        last_iteration = final_list
    return final_list
