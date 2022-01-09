import sys
sys.path.append("..")

from Commen_Functions import get_score, get_score_df, merge_Results, load_sets
import pickle
import pandas as pd
import numpy as np

# Script to load all gridsearch results and export them to a single pickle file
# This is in a single script because it takes hours to do this. This will make it runable as job

# Loop over all 4 sets
for x in range(1, 5):
    # Load the data
    df, df_normalized, Type_df = load_sets(x)
    
    results_grid = {}
    # Path to the different sets
    path = f'/home/g0017139/UMCG_Thesis/Scripts/Results/TSet{x}/'
    # Get the baseline score by putting the types as cluster labels in get scoree
    baseline = get_score(df_normalized, df['TYPE'].fillna('Unknown').values)
    # Set the output to a file for saving
    baseline_df = pd.DataFrame()
    baseline_df['n_clusters'] = np.arange(1, 31)
    baseline_df['algorithm'] = 'Baseline'
    baseline_df['silhouette_score_euclidean'] = baseline['silhouette_score_euclidean']
    baseline_df['silhouette_score_correlation'] = baseline['silhouette_score_correlation']
    # Save the baseline
    with open(f'{path}baseline_df.pkl', 'wb') as f:
        pickle.dump(baseline_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Load the hierachical clustering and calculate the scores
    clustersHier, results_grid['Hierarch'] = merge_Results(f'{path}',
                                                       'Hierarch', df_normalized)
    # Load dbscan and kmeans this score already has a silhouette so it doesn't need to be calculated
    with open(f"{path}kmeans.pkl", 'rb') as f:
        results_grid['kMeans'] = pickle.load(f)
    with open(f'{path}dbscan.pkl', 'rb') as f:
        results_grid['dbscan'] = pickle.load(f)
    # Load hdcc and calculate the scores
    clustersHDDC, results_grid['hddc'], bic_df = merge_Results(f'{path}',
                                                           'HDDC', df_normalized)
    # SAVE everything to pickle file for better better usability
    with open(f'{path}Merged_results.pkl', 'wb') as f:
        pickle.dump(results_grid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{path}Merged_GridHDDC.pkl', 'wb') as f:
        pickle.dump(bic_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{path}MergedClustersHDDC.pkl', 'wb') as f:
        pickle.dump(clustersHDDC, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{path}MergedClustersHierarch.pkl', 'wb') as f:
        pickle.dump(clustersHier, f, protocol=pickle.HIGHEST_PROTOCOL)