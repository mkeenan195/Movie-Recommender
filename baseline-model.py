#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Trains the baseline model
Usage:
    $ spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true baseline-model.py  splits-small output-small
    $ yarn logs -applicationId application_1648648882306_31254 -log_files stdout
'''

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, asc, desc, collect_list, array, lit, udf
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.evaluation import RankingEvaluator, RegressionEvaluator
from itertools import product
import pandas as pd


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

def evaluate_baseline(train_set, val_sets, val_raw, betas):
    # Create dictionary to store results
    results = {}
    for name, val_set in val_sets.items():
        results[name] = []

    # Loop through Betas, training and evaluating
    for beta in betas:
        base_pred = train_set.groupBy('movieId').agg({'userId':'count', 'rating':'sum'})
        base_pred = base_pred.withColumn('rating_pred', col('sum(rating)') / (col('count(userId)') + lit(beta)))
        base_recs = base_pred.sort('rating_pred', ascending=False).limit(100).select('movieId').rdd.flatMap(lambda x: x).collect()
        
        # Loop through main and alternative val sets
        for name, val_set in val_sets.items():
            val_pred = val_set.withColumn('predictions', array([lit(x) for x in base_recs]))
            val_pred = val_pred.withColumn("predictions",val_pred.predictions.cast(ArrayType(DoubleType())))

            evaluator = RankingEvaluator(predictionCol='predictions', labelCol='truth', k=100)
            meanap = evaluator.evaluate(val_pred, {evaluator.metricName: "meanAveragePrecisionAtK", evaluator.k: 100})
            ndcg = evaluator.evaluate(val_pred, {evaluator.metricName: "ndcgAtK", evaluator.k: 100})
            predictions = val_raw[name].join(base_pred, on='movieId') 
            #predictions.show(20)
            evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='rating_pred')
            rmse = evaluator.evaluate(predictions)
            results[name].append([beta, meanap, ndcg, rmse])
            print(name, beta, meanap, ndcg, rmse)
        
    return results




if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline-model').getOrCreate()

    # Folders for input and output data
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Read data
    train, val_holdout, test_holdout, val_holdout_four, test_holdout_four = read_datasets(spark, input_dir)

    # Evaluate over betas
    betas = [0, 5, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    val_sets = {'all_ratings':val_holdout, 'four_plus_ratings':val_holdout_four}
    val_raw = {'all_ratings':spark.read.parquet(input_dir+'/val_holdout.parquet'), 'four_plus_ratings':spark.read.parquet(input_dir+'/val_holdout_four.parquet')}
    beta_search = evaluate_baseline(train_set=train, val_sets=val_sets, val_raw=val_raw, betas=betas)
    print(beta_search)

    # Export hyperparameter search results
    def export_results(table, filename):
        table = pd.DataFrame(table, columns=['beta', 'map', 'ndcg', 'rmse'])
        table_sdf = spark.createDataFrame(table)
        table_sdf.coalesce(1).write.csv(output_dir+'/'+filename, mode='overwrite', header=True)
    export_results(beta_search['all_ratings'], 'baseline-model-hyperparameter-results.csv')
    export_results(beta_search['four_plus_ratings'], 'baseline-model-hyperparameter-results-four.csv')

    # Choose best model hyperparameters
    df = pd.DataFrame(beta_search['all_ratings'], columns=['beta', 'map', 'ndcg', 'rmse'])
    best_beta_map, _ , _, _ = df.iloc[df.map.argmax()]
    best_beta_ndcg, _ , _, _ = df.iloc[df.ndcg.argmax()]
    best_beta_rmse, _ , _, _ = df.iloc[df.rmse.argmin()]
    df = pd.DataFrame(beta_search['four_plus_ratings'], columns=['beta', 'map', 'ndcg', 'rmse'])
    best_beta_map_four, _ , _, _ = df.iloc[df.map.argmax()]
    best_beta_ndcg_four, _ , _, _ = df.iloc[df.ndcg.argmax()]
    best_beta_rmse_four, _ , _, _ = df.iloc[df.rmse.argmin()]

    # Evaluate on test set
    val_sets = {'all_ratings': test_holdout}
    val_raw = {'all_ratings': spark.read.parquet(input_dir+'/test_holdout.parquet')}
    betas = [best_beta_map, best_beta_ndcg, best_beta_rmse]
    test_results = evaluate_baseline(train_set=train, val_sets=val_sets, val_raw=val_raw, betas=betas)
    print(test_results)
    # Evaluate on test-four set
    val_sets = {'four_plus_ratings': test_holdout_four}
    val_raw = {'four_plus_ratings': spark.read.parquet(input_dir+'/test_holdout_four.parquet')}
    betas = [best_beta_map_four, best_beta_ndcg_four, best_beta_rmse_four]
    test_results_four = evaluate_baseline(train_set=train, val_sets=val_sets, val_raw=val_raw, betas=betas)
    print(test_results_four)

    # Export test set NDCG results
    export_results(test_results['all_ratings'], 'baseline-model-test-results.csv')
    export_results(test_results_four['four_plus_ratings'], 'baseline-model-test-results-four.csv')
    
