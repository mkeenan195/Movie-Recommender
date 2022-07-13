#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Trains the latent factor model with alternating least squares.
Usage:
    $ spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true train-test-split.py ml-latest-small splits-small
    $ yarn logs -applicationId application_1648648882306_31254 -log_files stdout
'''

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, asc, desc, collect_list, array, lit, udf, when
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RankingEvaluator, RegressionEvaluator
from itertools import product
import pandas as pd
import pickle

r_seed = 4321

def read_datasets(spark, input_dir):
    '''
    Inputs: 
        spark: spark session object
        input_dir: folder where train, val_holdout, and test_holdout parquet files are saved
    Returns:
        train: sparse format training data
        val_holdout: users' lists of heldout movies for hyperparameter tuning
        test_holdout: users' lists of heldout movies for final evaluation

    '''
    # Read parquet files
    train = spark.read.parquet(input_dir+'/train.parquet')
    val_holdout = spark.read.parquet(input_dir+'/val_holdout.parquet')
    test_holdout = spark.read.parquet(input_dir+'/test_holdout.parquet')
    val_holdout_four = spark.read.parquet(input_dir+'/val_holdout_four.parquet')
    test_holdout_four = spark.read.parquet(input_dir+'/test_holdout_four.parquet')

    # Transform validation set to have list of heldout movies
    val_holdout = val_holdout.groupby('userId').agg(collect_list('movieId'))
    val_holdout = val_holdout.withColumnRenamed("collect_list(movieId)","truth")
    val_holdout = val_holdout.withColumn("truth",val_holdout.truth.cast(ArrayType(DoubleType())))

    # Transform validation_four set to have list of heldout movies
    val_holdout_four = val_holdout_four.groupby('userId').agg(collect_list('movieId'))
    val_holdout_four = val_holdout_four.withColumnRenamed("collect_list(movieId)","truth")
    val_holdout_four = val_holdout_four.withColumn("truth",val_holdout_four.truth.cast(ArrayType(DoubleType())))

    # Transform test set to have list of heldout movies
    test_holdout = test_holdout.groupby('userId').agg(collect_list('movieId'))
    test_holdout = test_holdout.withColumnRenamed("collect_list(movieId)","truth")
    test_holdout = test_holdout.withColumn("truth",test_holdout.truth.cast(ArrayType(DoubleType())))

    # Transform test_four set to have list of heldout movies
    test_holdout_four = test_holdout_four.groupby('userId').agg(collect_list('movieId'))
    test_holdout_four = test_holdout_four.withColumnRenamed("collect_list(movieId)","truth")
    test_holdout_four = test_holdout_four.withColumn("truth",test_holdout_four.truth.cast(ArrayType(DoubleType())))

    return train, val_holdout, test_holdout, val_holdout_four, test_holdout_four

def evaluate_model(train_set, val_sets, val_raw, ranks, regParams, savePath=None):
    '''
    Inputs:
        train_set: sparse format training data
        val_set: heldout movies for hyperparameter tuning
        ranks: rank hyperparameter options to search over
        regParams: regParam hyperparameter options to search over
    Returns:
        results: array of hyperparamters and NDCG results on validation set
    '''

    # List to collect NDCG results
    results = {}
    for name, val_set in val_sets.items():
        results[name] = []

    # Function for converting recommendations to a list
    pred_to_list = udf(lambda x: [float(a.movieId) for a in x], ArrayType(DoubleType()))

    # Loop over all hyperparameter combinations
    hyper_parameters = product(ranks, regParams)
    for hyper_parameter in hyper_parameters:
        rank, regParam = hyper_parameter

        # Create model
        als = ALS(
            maxIter=50,  
            rank=rank, 
            regParam=regParam, 
            userCol="userId", 
            itemCol="movieId", 
            ratingCol="rating",
            coldStartStrategy="drop",
            numUserBlocks=10,
            numItemBlocks=10,
            seed = r_seed,
            checkpointInterval=5,
        )

        # Train model
        model = als.fit(train_set)
        if savePath != None:
            itemFactors = model.itemFactors
            userFactors = model.userFactors
            itemFactors = itemFactors.withColumn("features", col("features").cast("string"))
            userFactors = userFactors.withColumn("features", col("features").cast("string"))
            itemFactors.coalesce(1).write.csv(output_dir+'/itemFactors.csv', mode='overwrite', header=True)
            userFactors.coalesce(1).write.csv(output_dir+'/userFactors.csv', mode='overwrite', header=True)

        # Evaluate on heldout validation data and save results
        userRecs = model.recommendForUserSubset(dataset=val_set, numItems=100) # val_set is from loop above
        userRecs = userRecs.withColumn('recommendations',pred_to_list(col('recommendations'))).toDF('userId','predictions')
        for name, val_set in val_sets.items():
            # MAP and NDCG
            userRecsVal = userRecs.join(val_set, on='userId')
            evaluator = RankingEvaluator(predictionCol='predictions', labelCol='truth', k=100)
            meanap = evaluator.evaluate(userRecsVal, {evaluator.metricName: "meanAveragePrecisionAtK", evaluator.k: 100})
            ndcg = evaluator.evaluate(userRecsVal, {evaluator.metricName: "ndcgAtK", evaluator.k: 100})
            # RMSE
            predictions = model.transform(val_raw[name])
            evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
            rmse = evaluator.evaluate(predictions)
            results[name].append([rank, regParam, meanap, ndcg, rmse])
            print(name, rank, regParam, meanap, ndcg, rmse)

    return results

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('als-model').getOrCreate()
    spark.sparkContext.setCheckpointDir('checkpoints')

    # Folders for input and output data
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Read in the data
    train, val_holdout, test_holdout, val_holdout_four, test_holdout_four = read_datasets(spark, input_dir)

    # Evaluate model over given hyperparameters
    val_sets = {'all_ratings': val_holdout, 'four_plus_ratings':val_holdout_four}
    val_raw = {'all_ratings':spark.read.parquet(input_dir+'/val_holdout.parquet'), 'four_plus_ratings':spark.read.parquet(input_dir+'/val_holdout_four.parquet')}
    hp_search = evaluate_model(
        train_set = train, 
        val_sets = val_sets, 
        val_raw = val_raw,
        ranks = [10, 75, 200], 
        regParams = [0.001, 0.005, 0.01, 0.05, 0.1],
    )
    print('res', hp_search)

    # Export hyperparameter search results
    def export_results(table, filename):
        table = pd.DataFrame(table, columns=['rank', 'regParam', 'map', 'ndcg', 'rmse'])
        table_sdf = spark.createDataFrame(table)
        table_sdf.coalesce(1).write.csv(output_dir+'/'+filename, mode='overwrite', header=True)
    export_results(hp_search['all_ratings'], 'als-model-hyperparameter-results.csv')
    export_results(hp_search['four_plus_ratings'], 'als-model-hyperparameter-results-four.csv')
    
    # Choose best model hyperparameters
    df = pd.DataFrame(hp_search['all_ratings'], columns=['rank', 'regParam', 'map', 'ndcg', 'rmse'])
    best_rank_map, best_regParam_map, _, _, _ = df.iloc[df.map.argmax()]
    best_rank_ndcg, best_regParam_ndcg, _, _, _ = df.iloc[df.ndcg.argmax()]
    best_rank_rmse, best_regParam_rmse, _, _, _ = df.iloc[df.rmse.argmin()]
    best_hp = [[best_rank_map, best_regParam_map],
               [best_rank_ndcg, best_regParam_ndcg], 
               [best_rank_rmse, best_regParam_rmse]]
    df = pd.DataFrame(hp_search['four_plus_ratings'], columns=['rank', 'regParam', 'map', 'ndcg', 'rmse'])
    best_rank_map_four, best_regParam_map_four, _, _, _ = df.iloc[df.map.argmax()]
    best_rank_ndcg_four, best_regParam_ndcg_four, _, _, _ = df.iloc[df.ndcg.argmax()]
    best_rank_rmse_four, best_regParam_rmse_four, _, _, _ = df.iloc[df.rmse.argmin()]
    best_hp_four = [[best_rank_map_four, best_regParam_map_four],
                    [best_rank_ndcg_four, best_regParam_ndcg_four], 
                    [best_rank_rmse_four, best_regParam_rmse_four]]

    # Evaluate on test set
    val_sets = {'all_ratings': test_holdout}
    val_raw = {'all_ratings': spark.read.parquet(input_dir+'/test_holdout.parquet')}
    test_results = []
    for rank, regParam in best_hp:
        results = evaluate_model(
            train_set = train, 
            val_sets = val_sets, 
            val_raw = val_raw,
            ranks = [rank], 
            regParams = [regParam],
            savePath = output_dir
        )
        test_results.append(results['all_ratings'][0])
    print('test', test_results)
    export_results(test_results, 'als-model-test-results.csv')

    # Evaluate on test-four set
    val_sets = {'four_plus_ratings': test_holdout_four}
    val_raw = {'four_plus_ratings': spark.read.parquet(input_dir+'/test_holdout_four.parquet')}
    test_results_four = []
    for rank, regParam in best_hp_four:
        results = evaluate_model(
            train_set = train, 
            val_sets = val_sets, 
            val_raw = val_raw,
            ranks = [rank], 
            regParams = [regParam],
        )
        test_results_four.append(results['four_plus_ratings'][0])
    print('test_four', test_results_four)
    export_results(test_results_four, 'als-model-test-results-four.csv')



