from Commen_Functions import get_score, get_score_df, merge_Results, load_sets
import pickle
import pandas as pd
import numpy as np


#LOAD
for x in range(4,5):
    df, df_normalized, Type_df = load_sets(x)
    
    results_grid = {}
    path = f'/home/g0017139/UMCG_Thesis/Working_Code/Results/TSet{x}/'
    baseline = get_score(df_normalized, df['TYPE'].fillna('Unknown').values)
    baseline_df = pd.DataFrame()
    baseline_df['n_clusters'] = np.arange(1, 31)
    baseline_df['algorithm'] = 'Baseline'
    baseline_df['silhouette_score_euclidean'] = baseline['silhouette_score_euclidean']
    baseline_df['silhouette_score_correlation'] = baseline['silhouette_score_correlation']
    with open(f'{path}baseline_df.pkl', 'wb') as f:
        pickle.dump(baseline_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    clustersHier, results_grid['Hierarch'] = merge_Results(f'{path}',
                                                       'Hierarch', df_normalized)
    with open(f"{path}kmeans.pkl", 'rb') as f:
        results_grid['kMeans'] = pickle.load(f)
    with open(f'{path}dbscan.pkl', 'rb') as f:
        results_grid['dbscan'] = pickle.load(f)
    clustersHDDC, results_grid['hddc'], bic_df = merge_Results(f'{path}',
                                                           'HDDC', df_normalized)
    # SAVE
    with open(f'{path}Merged_results.pkl', 'wb') as f:
        pickle.dump(results_grid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{path}Merged_GridHDDC.pkl', 'wb') as f:
        pickle.dump(bic_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{path}MergedClustersHDDC.pkl', 'wb') as f:
        pickle.dump(clustersHDDC, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{path}MergedClustersHierarch.pkl', 'wb') as f:
        pickle.dump(clustersHier, f, protocol=pickle.HIGHEST_PROTOCOL)