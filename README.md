## NYU DSGA-1004 - Big Data, Final Project
#### Group 29: Martin Keenan

This repository contains the final report and all supporting code for the Big Data final project. 

### Instructions to run code
Clone this directory on the Peel cluster and run 

```console
source shell_setup.sh
```

Unzip the small and large datasets into folders named "ml-latest-small" and "ml-latest." The datasets can be found at https://grouplens.org/datasets/movielens/latest/.

To create the training, validation, and test sets from the small dataset, run the following command with the relevant arguments.
```console
spark-submit 
--conf  spark.dynamicAllocation.enabled=true 
--conf spark.shuffle.service.enabled=false
--conf spark.dynamicAllocation.shuffleTracking.enabled=true 
train-test-split.py [ml-latest-small/ml-latest] [splits-small/splits] 0.6 0.2 0.2
```

To run the baseline and latent factor models on the small and large datasets, run the following command with the relevant arguments.
```console
spark-submit 
--conf  spark.dynamicAllocation.enabled=true 
--conf spark.shuffle.service.enabled=false
--conf spark.dynamicAllocation.shuffleTracking.enabled=true 
[baseline-model.py/als-model.py] [splits-small/splits] [output-small/output]
```

After the models have been trained and evaluated, download the results and latent factors from Peel.

ScaNN is only supported on Linux environments. If using Windows or MacOS, consider building a [Docker container](https://hub.docker.com/_/python) with Python 3.8, installing [ScaNN](https://github.com/google-research/google-research/tree/master/scann), and running the scann-extension.py file.
