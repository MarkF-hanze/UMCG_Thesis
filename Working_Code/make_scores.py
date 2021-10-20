from Commen_Funtions import get_score, get_score_df, merge_Results, load_sets
import pickle


#LOAD
for x in range(1,5)
    df, df_normalized, Type_df = load_sets(x)
    results_grid = {}
    clustersHier, results_grid['Hierarch'] = merge_Results('/home/g0017139/UMCG_Thesis/Working_Code/Results/Set1/',
                                                       'Hierarch', df_normalized)
    with open("/home/g0017139/UMCG_Thesis/Working_Code/Results/Set1/kmeans.pkl", 'rb') as f:
        results_grid['kMeans'] = pickle.load(f)
    with open('/home/g0017139/UMCG_Thesis/Working_Code/Results/Set1/dbscan.pkl', 'rb') as f:
        results_grid['dbscan'] = pickle.load(f)
    clustersHDDC, results_grid['hddc'], bic_df = merge_Results('/home/g0017139/UMCG_Thesis/Working_Code/Results/Set1/',
                                                           'HDDC', df_normalized)
    # SAVE
    with open('/home/g0017139/UMCG_Thesis/Working_Code/Results/Set1/Merged_results.pkl', 'wb') as f:
        pickle.dump(results_grid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/home/g0017139/UMCG_Thesis/Working_Code/Results/Set1/Merged_GridHDDC.pkl', 'wb') as f:
        pickle.dump(bic_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/home/g0017139/UMCG_Thesis/Working_Code/Results/Set1/MergedClustersHDDC.pkl', 'wb') as f:
        pickle.dump(clustersHDDC, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/home/g0017139/UMCG_Thesis/Working_Code/Results/Set1/MergedClustersHierarch.pkl', 'wb') as f:
        pickle.dump(clustersHier, f, protocol=pickle.HIGHEST_PROTOCOL)