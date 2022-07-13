#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Split the dataset into training, validation and test observations.
Usage:
    $ spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true train-test-split.py ml-latest-small splits-small
'''

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, rand, when, row_number, desc
from pyspark.sql.window import Window

# Set random seed
r_seed = 4321

def main(spark, input_dir, output_dir, train_perc, val_perc, test_perc):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object
    '''

    # Read ratings data
    input_data = input_dir+'/ratings.csv'
    ratings = spark.read.csv(
        input_data, 
        header=True,
        schema='userId INT, movieId INT, rating FLOAT, timestamp LONG'
    )
    ratings.createOrReplaceTempView('ratings')

    # Split users into 50/20/30 train/val/test groups
    unique_users = spark.sql('SELECT DISTINCT userId FROM ratings ORDER BY userId ASC')
    train_users, val_users, test_users = unique_users.randomSplit(weights=[train_perc,val_perc,test_perc], seed=r_seed)
    train_users.createOrReplaceTempView('train_users')
    val_users.createOrReplaceTempView('val_users')
    test_users.createOrReplaceTempView('test_users')

    # Create dataset for each user group
    train = spark.sql('SELECT ratings.userId AS userId, movieId, rating, timestamp FROM train_users JOIN ratings ON train_users.userId=ratings.userId')
    val = spark.sql('SELECT ratings.userId AS userId, movieId, rating, timestamp FROM val_users JOIN ratings ON val_users.userId=ratings.userId')
    test = spark.sql('SELECT ratings.userId AS userId, movieId, rating, timestamp FROM test_users JOIN ratings ON test_users.userId=ratings.userId')
    train.createOrReplaceTempView('train')
    val.createOrReplaceTempView('val')
    test.createOrReplaceTempView('test')

    # Count number of ratings for each user
    test = test.join(spark.sql('SELECT userId, COUNT(movieId) AS total_movies FROM test GROUP BY userId'), how='left', on='userId')
    val = val.join(spark.sql('SELECT userId, COUNT(movieId) AS total_movies FROM val GROUP BY userId'), how='left', on='userId')

    # Choose the 30% most recent reviews from each val/test user as holdout observations
    test = test.withColumn("rank", row_number().over(Window.partitionBy("userId").orderBy(desc("timestamp"))))
    val = val.withColumn("rank", row_number().over(Window.partitionBy("userId").orderBy(desc("timestamp"))))
    test = test.withColumn("holdout", when(test.rank / test.total_movies <= 0.30, lit(1)).otherwise(lit(0)))
    val = val.withColumn("holdout", when(val.rank / val.total_movies <= 0.30, lit(1)).otherwise(lit(0)))

    # Separate the training and holdout portions of the test and validation sets
    test_holdout = test.filter(test.holdout==1).drop('rand','rank','total_movies','holdout')
    test_train = test.filter(test.holdout==0).drop('rand','rank','total_movies','holdout')
    val_holdout = val.filter(val.holdout==1).drop('rand','rank','total_movies','holdout')
    val_train = val.filter(val.holdout==0).drop('rand','rank','total_movies','holdout')

    # Add group column to differentiate user groups after combining
    train = train.withColumn('group', lit('train'))
    val_train = val_train.withColumn('group', lit('val_train'))
    test_train = test_train.withColumn('group', lit('test_train'))
    val_holdout = val_holdout.withColumn('group', lit('val_holdout'))
    test_holdout = test_holdout.withColumn('group', lit('test_holdout'))

    # Combine all the observations that will be used for training
    train = train.union(val_train).union(test_train)

    # Write train, val holdout and test holdout data to parquet
    train.write.parquet(output_dir+'/train.parquet')
    val_holdout.write.parquet(output_dir+'/val_holdout.parquet')
    test_holdout.write.parquet(output_dir+'/test_holdout.parquet')

    # Create holdout sets that only contain ratings of four or greater
    val_holdout = val_holdout.filter(val_holdout.rating>=4.0)
    test_holdout = test_holdout.filter(test_holdout.rating>=4.0)
    val_holdout.write.parquet(output_dir+'/val_holdout_four.parquet')
    test_holdout.write.parquet(output_dir+'/test_holdout_four.parquet')


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('train-test-split').getOrCreate()

    # Folders for input and output data and shares for train, val, and test users
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    train_perc = float(sys.argv[3])
    val_perc = float(sys.argv[4])
    test_perc = float(sys.argv[5])

    # run main function
    main(spark, input_dir, output_dir, train_perc, val_perc, test_perc)
    
