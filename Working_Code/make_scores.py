from Commen_Functions import get_score, get_score_df, merge_Results, load_sets
import pickle


#LOAD
for x in range(1,2):
    df, df_normalized, Type_df = load_sets(x)
    results_grid = {}
    path = f'/home/g0017139/UMCG_Thesis/Working_Code/Results/TSet{x}/'
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