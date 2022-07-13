import numpy as np
import pandas as pd
import scann
import time
import matplotlib.pyplot as plt

'''
This program compares the speed and recall of the ScaNN method for different tree partition structures.
'''

def convert_str_to_list(t):
    '''
    Convert string output of Spark to list.
    '''
    split_list = t.replace('[','').replace(']','').split(', ')
    return list(map(lambda x: float(x), split_list))

def load_data(input_dir):
    '''
    Load the user and item factors and convert to np array
    '''
    user = pd.read_csv(input_dir+'/userFactors.csv', nrows=500).sort_values(by='id').reset_index(drop=True)
    item = pd.read_csv(input_dir+'/itemFactors.csv').sort_values(by='id').reset_index(drop=True)
    user.features = user.features.apply(convert_str_to_list)
    item.features = item.features.apply(convert_str_to_list)
    userFactors = np.array(user.features.values.tolist())
    itemFactors = np.array(item.features.values.tolist())
    print('User Shape: ', userFactors.shape)
    print('Item Shape: ', itemFactors.shape) 
    return userFactors, itemFactors

def compute_recall(calc_neighbors, true_neighbors):
    '''
    Compute recall between approximate recommendations and true recommendations
    '''
    total = 0
    for tr, r in zip(true_neighbors, calc_neighbors):
        total += np.intersect1d(tr, r).shape[0]
    return total / true_neighbors.size

def numpy_recommend(num_queries, userFactors, itemFactors, k):
    '''
    Brute force similarity search with numpy
    '''
    dot = itemFactors @ userFactors[:num_queries].T
    neighbors = np.argpartition(dot, -k, axis=0)[-k:].T
    distances = np.partition(dot, -k, axis=0)[-k:].T
    return neighbors, distances

def time_numpy(num_queries, userFactors, itemFactors, k, num_loops=100):
    '''
    Time for brute force method.
    '''
    t = 0
    for loop in range(num_loops):
        start = time.time()
        np_neighbors, np_distances = numpy_recommend(num_queries, userFactors, itemFactors, 100)
        end = time.time()
        t += end - start
    np_time = t / num_loops
    np_qps = num_queries / np_time
    return np_neighbors, np_qps

def time_scann(searcher, userFactors, num_queries, num_loops):
    '''
    Time for scann method.
    '''
    t = 0
    for loop in range(num_loops):
        start = time.time()
        scann_neighbors, scann_distances = searcher.search_batched(userFactors[:num_queries])
        end = time.time()
        t += end - start
    mean_time = t / num_loops
    return mean_time, scann_neighbors

def implement_scann(itemFactors, userFactors, true_neighbors, num_queries, num_leaves_grid=[5,10,50], search_proportion=0.10, rescoring=True, brute_force=False, num_loops=100):
    '''
    Implement ScaNN search.
    '''
    seconds = []
    recalls = []        
    for num_leaves in num_leaves_grid:
        print(num_leaves)
        if brute_force == True:
            searcher = scann.scann_ops_pybind.builder(itemFactors, 100, "dot_product").score_brute_force().build()
        elif rescoring == True:
            searcher = scann.scann_ops_pybind.builder(itemFactors, 100, "dot_product").tree(
                num_leaves=num_leaves, num_leaves_to_search=int(num_leaves*search_proportion)).score_ah(
                dimensions_per_block=2, anisotropic_quantization_threshold=0.2).reorder(1000).build()
        else:
            searcher = scann.scann_ops_pybind.builder(itemFactors, 100, "dot_product").tree(
                num_leaves=num_leaves, num_leaves_to_search=int(num_leaves*search_proportion)).score_ah(
                dimensions_per_block=2, anisotropic_quantization_threshold=0.2).build()
        
        mean_time, scann_neighbors = time_scann(searcher, userFactors, num_queries, num_loops=num_loops)

        recall = compute_recall(scann_neighbors, true_neighbors)
        print(recall, mean_time)
        seconds.append(mean_time)
        recalls.append(recall)   
    qps = list(map(lambda x: num_queries / x, seconds))
    return qps, recalls

def plot_scann(eff_res, np_qps):
    '''
    Plot queries per second and recall results
    '''
    fig, ax = plt.subplots()
    ax.plot(eff_res[0.05][1], eff_res[0.05][0], marker='x', label='5%')
    ax.plot(eff_res[0.10][1], eff_res[0.10][0], marker='x', label='10%')
    ax.plot(eff_res[0.20][1], eff_res[0.20][0], marker='x', label='20%')
    ax.scatter([1.0], [np_qps], color='red', label='Numpy Brute Force')
    ax.set_yscale('log')
    ax.yaxis.label.set_text('Queries per Second')
    ax.xaxis.label.set_text('Recall')
    plt.legend()
    plt.show()
    return
    
def export_results(output_dir, eff_res, np_qps):
    '''
    Export results to CSV.
    '''
    df = pd.DataFrame()
    df['qps05'] = eff_res[0.05][0]
    df['rec05'] = eff_res[0.05][1]
    df['qps10'] = eff_res[0.10][0]
    df['rec10'] = eff_res[0.10][1]
    df['qps20'] = eff_res[0.20][0]
    df['rec20'] = eff_res[0.20][1]
    df['brute'] = np_qps
    df.to_csv(output_dir+'/scann_times.csv')
    return
    
if __name__ == '__main__':

    ### Small Dataset
    # Read 500 rows of users and all of movies
    userFactors, itemFactors = load_data('output-small')
    # Set number of queries and number of partitions
    num_queries=500
    num_leaves_grid = [25, 250, 1_000, 2_500, 5_000]
    # Time brute force method
    np_neighbors, np_qps = time_numpy(num_queries, userFactors, itemFactors, k=100, num_loops=10)
    # Time ScaNN method
    eff_res = {}
    for prop in [0.05, 0.10, 0.20]:
        eff_res[prop] = implement_scann(itemFactors, userFactors, np_neighbors, 
                                        num_queries, num_leaves_grid=num_leaves_grid,
                                        search_proportion=prop, rescoring=True, 
                                        brute_force=False, num_loops=10)
    # Plot and save results
    plot_scann(eff_res, np_qps)
    export_results('output-small', eff_res, np_qps)


    ## Large Dataset
    # Read 500 rows of users and all of movies
    userFactors, itemFactors = load_data('output')
    # Set number of queries and number of partitions
    num_queries=500
    num_leaves_grid = [250, 2_500, 10_000, 25_000, 50_000]
    # Time brute force method
    np_neighbors, np_qps = time_numpy(num_queries, userFactors, itemFactors, k=100, num_loops=10)
    # Time ScaNN method
    eff_res = {}
    for prop in [0.05, 0.10, 0.20]:
        eff_res[prop] = implement_scann(itemFactors, userFactors, np_neighbors, 
                                        num_queries, num_leaves_grid=num_leaves_grid,
                                        search_proportion=prop, rescoring=True, 
                                        brute_force=False, num_loops=10)
    # Plot and save results
    plot_scann(eff_res, np_qps)
    export_results('output', eff_res, np_qps)
